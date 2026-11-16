.. _forward_mode_jets:

Forward-Mode Jets
=================

.. currentmodule:: warp

.. admonition:: Experimental

    Forward-mode jets are experimental and subject to change without a formal deprecation cycle.

Warp's :class:`wp.Tape <Tape>` and :func:`wp.grad <grad>` differentiate in reverse mode: they replay a computation
backwards and deliver the gradient of one output with respect to every input in a single sweep (see
:ref:`differentiability`). That is the right tool for a loss over a large state, but it has limitations. For one,
reverse mode in Warp currently cannot be applied to its own output, so on its own it is limited to first derivatives.
For another, reverse mode needs a backward sweep for each output it differentiates, so it scales poorly for Jacobian
matrices of many-output functions and for Jacobian-vector products.

Forward-mode *jets* fill these gaps. A jet carries a value together with its derivatives along a fixed number of
directions. A :func:`@wp.func <func>` written generically over its argument types can be evaluated over jets instead
of floats, returning its derivatives along with its value, with no hand-written gradient or Hessian:

.. math::

    x = \text{value} + \sum_{q=1}^{w} \text{grad}_q \, \varepsilon_q

where :math:`w` is the jet's *width* and each :math:`\varepsilon_q` is a formal infinitesimal, one per direction,
whose products vanish: :math:`\varepsilon_q \varepsilon_r = 0` for all :math:`q, r`. Arithmetic on jets therefore keeps
only first-order terms. For example, multiplying two jets multiplies their values and combines their gradients by
the product rule, so :math:`\text{grad}_q` of any result is its derivative along direction :math:`q`.
A jet's ``grad`` field therefore holds the gradient of its value with respect to the :math:`w` directions, with one row
per component for vector-valued jets: the usual gradient or Jacobian when the jets are seeded with the coordinate
directions, and one directional derivative per direction otherwise.
:func:`wp.JetSpace <JetSpace>` creates these first-order jets. :func:`wp.JetSpace2 <JetSpace2>` creates second-order
jets, which keep the products :math:`\varepsilon_q \varepsilon_r` (dropping only third-order terms) and so also carry
a dense Hessian.

Jets do not replace reverse mode but complement it, with different trade-offs in memory access, operation count and
compile time. See :ref:`jets_when_to_use` below.

Three levels are involved:

- A *jet space*, created by :func:`wp.JetSpace <JetSpace>`, fixes one set of derivative directions. It returns a
  namespace, called ``J`` below, which holds a family of compatible jet types sharing those directions, along with
  seeding helpers such as ``J.seed()``.
- A *jet type* exists for each kind of value: ``J.scalar``, ``J.vec2``, ``J.mat22`` (a jet whose value is a 2x2
  matrix), and so on. Each type is shared by every input and intermediate of that kind, and is not tied to any
  particular input.
- A *jet* is an instance of one of these types, created inside a kernel by the seeding helpers. A jet ``a`` holds its
  value in ``a.value`` and its derivative data in ``a.grad``. The namespace also names the plain Warp types of that
  gradient data: for example, the gradients of a ``J.vec2`` jet are a ``J.grad_type_vec2``, an ordinary
  :math:`2 \times w` matrix whose rows are the derivatives of the two components.

A Small First Example
---------------------

Take :math:`u = a + b` and :math:`v = a - b` at :math:`a = 2, b = 3`. With width 2, ``Jcoord.seed()`` seeds
each input along its own direction, so the gradients of ``a`` and ``b`` are the rows of the identity, and the
gradients of ``u`` and ``v`` are the rows of the Jacobian. With width 1 and the single direction :math:`(1, -1)`,
both inputs vary along one shared direction, and the gradients of ``u`` and ``v`` are the Jacobian times that
direction:

.. testcode::

    Jcoord = wp.JetSpace((wp.float32, wp.float32))  # width 2: one direction per input
    Jdir = wp.JetSpace((wp.float32, wp.float32), width=1)  # width 1: one chosen direction

    @wp.kernel
    def sum_difference_kernel(
        jacobian: wp.array[Jcoord.grad_type_scalar],
        jacobian_times_direction: wp.array[Jdir.grad_type_scalar],
    ):
        # Coordinate directions: a.grad = (1, 0), b.grad = (0, 1).
        a, b = Jcoord.seed(2.0, 3.0)
        u = a + b
        v = a - b
        jacobian[0] = u.grad
        jacobian[1] = v.grad

        # One direction (1, -1): a.grad = (1), b.grad = (-1).
        a1, b1 = Jdir.directional(2.0, 3.0, 1.0, -1.0)
        u1 = a1 + b1
        v1 = a1 - b1
        jacobian_times_direction[0] = u1.grad
        jacobian_times_direction[1] = v1.grad

    jacobian = wp.zeros(2, dtype=Jcoord.grad_type_scalar)
    jacobian_times_direction = wp.zeros(2, dtype=Jdir.grad_type_scalar)
    wp.launch(sum_difference_kernel, dim=1, outputs=[jacobian, jacobian_times_direction])

    print(f"jacobian =\n{jacobian.numpy()}")
    print("jacobian_times_direction =", jacobian_times_direction.numpy()[:, 0])

.. testoutput::

    jacobian =
    [[ 1.  1.]
     [ 1. -1.]]
    jacobian_times_direction = [0. 2.]

Both are the same mechanism: the width is the number of directions, not the number of inputs, and coordinate seeding
is the special case where the directions are the coordinate axes.

Inverse Kinematics Example
--------------------------

Consider a two-link arm in 2D with four degrees of freedom: the position of its base ``base_pos``, the relative
shoulder angle ``shoulder`` and the relative elbow angle ``elbow``. A jet space for these inputs tracks derivatives
along each of them, with a width equal to their four scalar components. For inverse kinematics, the goal is to bring the
end of the arm to a target position ``target_pos``. The residual of that least-squares problem is an ordinary function,
written once and generic over its argument types, so it accepts both plain values and jets. A kernel or another
:func:`@wp.func <func>` can then seed the inputs, call ``residual``, and read off the gradients to construct the
residual's Jacobian:

.. testcode::

    J = wp.JetSpace((wp.vec2, wp.float32, wp.float32))  # width 4

    @wp.func
    def residual(base_pos: Any, shoulder: Any, elbow: Any, target_pos: wp.vec2):
        # Two links of unit length; the elbow angle is relative to the upper arm.
        elbow_pos = base_pos + wp.cos(shoulder) * wp.vec2(1.0, 0.0) + wp.sin(shoulder) * wp.vec2(0.0, 1.0)
        forearm = shoulder + elbow
        end_pos = elbow_pos + wp.cos(forearm) * wp.vec2(1.0, 0.0) + wp.sin(forearm) * wp.vec2(0.0, 1.0)
        return end_pos - target_pos

    @wp.kernel
    def residual_jacobian_kernel(
        base_pos: wp.array[wp.vec2],
        shoulder: wp.array[float],
        elbow: wp.array[float],
        target_pos: wp.array[wp.vec2],
        jacobian: wp.array[J.grad_type_vec2],  # a vec2 jet's gradients: 2x4 here
    ):
        tid = wp.tid()
        jet_base_pos, jet_shoulder, jet_elbow = J.seed(base_pos[tid], shoulder[tid], elbow[tid])
        jet_residual = residual(jet_base_pos, jet_shoulder, jet_elbow, target_pos[tid])
        jacobian[tid] = jet_residual.grad  # 2x4: d(residual) / d(base_pos, shoulder, elbow)

    base_pos = wp.array([[0.0, 0.0]], dtype=wp.vec2)
    shoulder = wp.array([np.radians(45.0)], dtype=float)
    elbow = wp.array([np.radians(-45.0)], dtype=float)
    target_pos = wp.array([[-0.5, 0.5]], dtype=wp.vec2)
    jacobian = wp.zeros(1, dtype=J.grad_type_vec2)

    wp.launch(residual_jacobian_kernel, dim=1, inputs=[base_pos, shoulder, elbow, target_pos], outputs=[jacobian])

    print(f"jacobian =\n{np.round(jacobian.numpy()[0], 4)}")

.. testoutput::

    jacobian =
    [[ 1.      0.     -0.7071  0.    ]
     [ 0.      1.      1.7071  1.    ]]

Nothing is bound into the module beyond the space ``J``. Operators and builtins such as ``end_pos - target_pos`` and
``wp.sin()`` resolve on jets because the jet arithmetic is registered as overloads of Warp's own builtins.

Slots and Seeding
-----------------

The default use case for jets is constructing the Jacobian of a function with respect to a predetermined set of
inputs. The :func:`wp.JetSpace <JetSpace>` constructor takes a tuple of types, called *slots*, such as
``(wp.vec2, wp.float32, wp.float32)`` in the example above. Each slot corresponds to one of the function's inputs. By
default, the width of the jets is the total number of scalar components (``width=4`` in the example above), so the jets
carry the full Jacobian with respect to these inputs.

The slot types are scalars, ``vec2``, ``vec3``, and the ``mat22``, ``mat33``, ``mat32`` and ``mat23`` matrices.
:func:`wp.JetSpace <JetSpace>` returns a namespace of types that share the same derivative directions. For each slot
type ``X`` it holds a jet type ``J.X`` (``J.scalar``, ``J.vec2``, ..., ``J.mat23``) and the type of that jet's
``grad`` field, ``J.grad_type_X``. For example, a ``J.vec2`` jet's gradients are a ``J.grad_type_vec2``, a
:math:`2 \times w` matrix whose row :math:`r` holds the derivatives of component :math:`r`. Matrix jets flatten their
entries row-major, so ``J.grad_type_mat22`` has four rows. Every input and intermediate of the same type shares one
jet type: ``J.vec2`` is the type of every ``vec2`` jet in the space, not of a particular slot. The same types can be
looked up by the plain Warp type, as in ``J.jet_type(wp.vec2)`` and ``J.grad_type(wp.vec2)``. These lookups work in
annotations and host code, but cannot be called inside a kernel or :func:`@wp.func <func>`, where the attributes are
needed.

Individual jets are instances of these types, and ``a.value`` and ``a.grad`` are the data a jet ``a`` holds. Inside a
kernel, ``J.seed()`` creates them: it takes one value per slot and returns one jet per slot. It copies each input's
value into the jet's ``value`` field and sets the ``grad`` field to the unit vector along the corresponding direction
of the derivative space, ordered by slot and then by component within the slot (row-major for matrices). In effect, the
jets are seeded with the identity matrix, so the gradients of any downstream output contain its Jacobian.

Width and Directional
---------------------

Jets can also compute directional derivatives, also known as Jacobian-vector products, without first constructing the
full Jacobian. To do this, construct :func:`wp.JetSpace <JetSpace>` with a ``width`` equal to the number of directions
to differentiate along, which does not need to match the total number of slot components. When the width differs from
the number of slot components, ``J.seed()`` is not available, and the derivative directions are provided explicitly
instead.

``J.directional()`` takes the same values as ``J.seed()``, followed by ``width * len(slots)`` direction values, ordered
direction-major: for each direction in turn, one value per slot, of the same type as that slot. The returned jets
carry these directions in their ``grad`` fields instead of the identity matrix, so the gradients of any downstream
output contain its Jacobian times the direction vectors.

In the example above, suppose we want the derivative of the residual as the shoulder rotates one way and the elbow
the other way at the same rate, so the forearm keeps its orientation. A space with scalar slots for ``shoulder`` and
``elbow`` and ``width=1`` (instead of the default ``2``) does this, given the direction ``(1, -1)``. The result is the
Jacobian times that direction, the shoulder column printed above minus the elbow column. Because the forearm only
translates, the end of the arm moves exactly as the elbow does:

.. testcode::

    Jd = wp.JetSpace((wp.float32, wp.float32), width=1)

    @wp.kernel
    def directional_derivative_kernel(
        base_pos: wp.array[wp.vec2],
        shoulder: wp.array[float],
        elbow: wp.array[float],
        target_pos: wp.array[wp.vec2],
        directional_derivative: wp.array[Jd.grad_type_vec2],
    ):
        tid = wp.tid()
        # Rotate the shoulder and the elbow in opposite directions at the same rate.
        jet_shoulder, jet_elbow = Jd.directional(shoulder[tid], elbow[tid], 1.0, -1.0)
        jet_residual = residual(base_pos[tid], jet_shoulder, jet_elbow, target_pos[tid])
        directional_derivative[tid] = jet_residual.grad

    directional_derivative = wp.zeros(1, dtype=Jd.grad_type_vec2)

    wp.launch(
        directional_derivative_kernel,
        dim=1,
        inputs=[base_pos, shoulder, elbow, target_pos],
        outputs=[directional_derivative],
    )

    print("directional_derivative =", np.round(directional_derivative.numpy()[0, :, 0], 4))

.. testoutput::

    directional_derivative = [-0.7071  0.7071]

Component-Wise Seeding
^^^^^^^^^^^^^^^^^^^^^^

Callers may also access the low-level interface of jets directly, without the slot types. Passing a number of scalars
instead of slot types, as in ``wp.JetSpace(4)``, creates a space whose width defaults to that number, or is the
``width`` provided. Each scalar variable is then seeded manually with ``J.variable(x, direction)``, which returns a
scalar jet with value ``x`` and a unit derivative along ``direction``.

.. testcode::

    Jn = wp.JetSpace(4)  # four scalars, width 4

    @wp.kernel
    def sum_of_cubes_gradient_kernel(x: wp.array2d[float], gradient: wp.array[Jn.grad_type_scalar]):
        tid = wp.tid()
        total = Jn.constant(0.0)
        for k in range(4):
            xk = Jn.variable(x[tid, k], k)
            total = total + xk * xk * xk
        gradient[tid] = total.grad

    x = wp.array([[1.0, 2.0, 3.0, 4.0]], dtype=float)
    gradient = wp.zeros(1, dtype=Jn.grad_type_scalar)

    wp.launch(sum_of_cubes_gradient_kernel, dim=1, inputs=[x], outputs=[gradient])

    print("gradient =", gradient.numpy()[0])

.. testoutput::

    gradient = [ 3. 12. 27. 48.]

Hessians
--------

To compute second derivatives, we can either combine Warp's reverse mode (:class:`wp.Tape <Tape>` or
:func:`wp.grad <grad>`) with first-order jets (:func:`wp.JetSpace <JetSpace>`), or use second-order jets
(:func:`wp.JetSpace2 <JetSpace2>`). For any of these methods discussed below, the function value and the gradient are
also computed as byproducts, so they can be used for Newton's method or other optimization algorithms.

:func:`wp.JetSpace2 <JetSpace2>` is constructed and seeded like :func:`wp.JetSpace <JetSpace>`, with scalar, ``vec2``
and ``vec3`` slots. A second-order scalar jet carries its value, its gradient ``grad`` and its dense Hessian
``hess``, whose types are ``J2.grad_type_scalar`` and ``J2.hess_type_scalar``:

.. testcode::

    J2 = wp.JetSpace2((wp.vec3, wp.vec3))  # width 6

    @wp.func
    def spring_energy(a: Any, b: Any):
        d = a - b
        return 0.5 * (wp.length(d) - 1.0) ** 2

    @wp.kernel
    def hessian_kernel(
        x0: wp.array[wp.vec3],
        x1: wp.array[wp.vec3],
        val: wp.array[float],
        grad: wp.array[J2.grad_type_scalar],
        hess: wp.array[J2.hess_type_scalar],
    ):
        i = wp.tid()
        jet_x0, jet_x1 = J2.seed(x0[i], x1[i])
        f = spring_energy(jet_x0, jet_x1)
        val[i] = f.value
        grad[i] = f.grad
        hess[i] = f.hess

    # A spring of rest length one, stretched to length two along x.
    x0 = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3)
    x1 = wp.array([[2.0, 0.0, 0.0]], dtype=wp.vec3)
    val = wp.zeros(1, dtype=float)
    grad = wp.zeros(1, dtype=J2.grad_type_scalar)
    hess = wp.zeros(1, dtype=J2.hess_type_scalar)

    wp.launch(hessian_kernel, dim=1, inputs=[x0, x1], outputs=[val, grad, hess])

    print("val =", val.numpy()[0])
    print("grad =", grad.numpy()[0])
    print(f"hess =\n{hess.numpy()[0]}")

.. testoutput::

    val = 0.5
    grad = [-1.  0.  0.  1.  0.  0.]
    hess =
    [[ 1.   0.   0.  -1.   0.   0. ]
     [ 0.   0.5  0.   0.  -0.5  0. ]
     [ 0.   0.   0.5  0.   0.  -0.5]
     [-1.   0.   0.   1.   0.   0. ]
     [ 0.  -0.5  0.   0.   0.5  0. ]
     [ 0.   0.  -0.5  0.   0.   0.5]]


Reverse-over-Forward
^^^^^^^^^^^^^^^^^^^^

For a small, fixed number of inputs, :func:`wp.JetSpace2 <JetSpace2>` is often the fastest way to compute a Hessian at
run time. Its state grows quadratically with the width, however, so it is register-intensive and can take a long time to
compile. For more inputs, or when shorter compile times matter, it can be better to combine Warp's reverse mode with
first-order jets: reverse over forward. The jet arithmetic is ordinary Warp code, so Warp's autodiff differentiates it,
including the seeding helpers, like any other function. This works either inside a kernel, with :func:`wp.grad <grad>`
applied to a :func:`@wp.func <func>`, or across kernels, by recording a :class:`wp.Tape <Tape>` and calling
``tape.backward()`` once per input component.

``wp.grad`` over ``wp.JetSpace``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`wp.grad <grad>` differentiates a :func:`@wp.func <func>`. A function that evaluates the energy over width-1 jets
along a direction :math:`v` returns the directional derivative :math:`\nabla E \cdot v`, and the gradient of that with
respect to the inputs is the Hessian-vector product :math:`H v`. With :math:`v` the unit vector :math:`e_j`, it is row
:math:`j` of the Hessian, so one call per input gives the full Hessian:

.. testcode::

    J1 = wp.JetSpace((wp.vec3, wp.vec3), width=1)

    @wp.func
    def directional_derivative(x0: wp.vec3, x1: wp.vec3, v0: wp.vec3, v1: wp.vec3):
        # A width-1 jet along the direction (v0, v1): the scalar dE . v.
        jet_x0, jet_x1 = J1.directional(x0, x1, v0, v1)
        return spring_energy(jet_x0, jet_x1).grad[0]

    @wp.kernel
    def grad_over_jet_hessian_kernel(
        x0: wp.array[wp.vec3],
        x1: wp.array[wp.vec3],
        hess: wp.array3d[float],
    ):
        i = wp.tid()
        for j in range(6):
            # The unit direction e_j over the six coordinates.
            v0 = wp.vec3(0.0)
            v1 = wp.vec3(0.0)
            if j < 3:
                v0[j] = 1.0
            else:
                v1[j - 3] = 1.0
            # The gradient of dE . e_j with respect to (x0, x1) is row j of the Hessian.
            hv0, hv1, _v0_adj, _v1_adj = wp.grad(directional_derivative)(x0[i], x1[i], v0, v1)
            for k in range(3):
                hess[i, j, k] = hv0[k]
                hess[i, j, 3 + k] = hv1[k]

    hess = wp.zeros((1, 6, 6), dtype=float)
    wp.launch(grad_over_jet_hessian_kernel, dim=1, inputs=[x0, x1], outputs=[hess])

    print(f"hess =\n{hess.numpy()[0]}")

.. testoutput::

    hess =
    [[ 1.   0.   0.  -1.   0.   0. ]
     [ 0.   0.5  0.   0.  -0.5  0. ]
     [ 0.   0.   0.5  0.   0.  -0.5]
     [-1.   0.   0.   1.   0.   0. ]
     [ 0.  -0.5  0.   0.   0.5  0. ]
     [ 0.   0.  -0.5  0.   0.   0.5]]

Each Hessian-vector product costs a constant multiple of one evaluation of the energy, so the full Hessian of :math:`n`
inputs costs :math:`O(n)` evaluations, against :math:`O(n^2)` for a dense second-order jet. In practice this route is
a bit slower than :func:`wp.JetSpace2 <JetSpace2>` at run time, with shorter compile times.

``wp.Tape`` over ``wp.JetSpace``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same directional derivative can be recorded on a :class:`wp.Tape <Tape>` instead. Each ``tape.backward()`` gives
one row of the Hessian as the adjoints of the inputs. Because of tape and kernel launch overhead, this is often the
slowest of the three methods, but it is more flexible than :func:`wp.grad <grad>` because the jets can pass between
kernel launches through arrays:

.. testcode::

    @wp.kernel
    def spring_directional_derivative_kernel(
        x0: wp.array[wp.vec3],
        x1: wp.array[wp.vec3],
        v0: wp.array[wp.vec3],
        v1: wp.array[wp.vec3],
        out: wp.array[float],
    ):
        i = wp.tid()
        out[i] = directional_derivative(x0[i], x1[i], v0[i], v1[i])

    x0 = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, requires_grad=True)
    x1 = wp.array([[2.0, 0.0, 0.0]], dtype=wp.vec3, requires_grad=True)
    out = wp.zeros(1, dtype=float, requires_grad=True)
    hess = np.zeros((1, 6, 6))

    tape = wp.Tape()
    for j in range(6):
        e = np.zeros((1, 6))
        e[:, j] = 1.0
        v0 = wp.array(e[:, :3], dtype=wp.vec3)
        v1 = wp.array(e[:, 3:], dtype=wp.vec3)
        with tape:
            wp.launch(spring_directional_derivative_kernel, dim=1, inputs=[x0, x1, v0, v1], outputs=[out])
        # The adjoints of x0 and x1 are row j of the Hessian.
        tape.backward(grads={out: wp.ones(1, dtype=float)})
        hess[:, j, :3] = x0.grad.numpy()
        hess[:, j, 3:] = x1.grad.numpy()
        tape.reset()

    print(f"hess =\n{hess[0]}")

.. testoutput::

    hess =
    [[ 1.   0.   0.  -1.   0.   0. ]
     [ 0.   0.5  0.   0.  -0.5  0. ]
     [ 0.   0.   0.5  0.   0.  -0.5]
     [-1.   0.   0.   1.   0.   0. ]
     [ 0.  -0.5  0.   0.   0.5  0. ]
     [ 0.   0.  -0.5  0.   0.   0.5]]

Hessian-Vector Products
~~~~~~~~~~~~~~~~~~~~~~~

Both reverse-over-forward methods build the Hessian one row at a time, as Hessian-vector products with unit vectors.
Seeding the width-1 jet with any other vector :math:`v` gives :math:`H v` from a single backward pass, without forming
the Hessian:

.. testcode::

    @wp.kernel
    def hessian_vector_product_kernel(
        x0: wp.array[wp.vec3],
        x1: wp.array[wp.vec3],
        v0: wp.array[wp.vec3],
        v1: wp.array[wp.vec3],
        hv: wp.array2d[wp.vec3],
    ):
        i = wp.tid()
        # The gradient of dE . v with respect to the inputs is H v.
        hv0, hv1, _v0_adj, _v1_adj = wp.grad(directional_derivative)(x0[i], x1[i], v0[i], v1[i])
        hv[i, 0] = hv0
        hv[i, 1] = hv1

    v0 = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3)
    v1 = wp.array([[1.0, 1.0, 0.0]], dtype=wp.vec3)
    hv = wp.zeros((1, 2), dtype=wp.vec3)

    wp.launch(hessian_vector_product_kernel, dim=1, inputs=[x0, x1, v0, v1], outputs=[hv])

    print("hv =", hv.numpy().reshape(6))

.. testoutput::

    hv = [-1.  -0.5  0.   1.   0.5  0. ]

.. _jets_when_to_use:

When to Use Jets
----------------

``width`` is a compile-time constant, not a runtime length. It becomes the length of ``grad`` and is unrolled into the
generated code, and every intermediate value carries all ``width`` components whether or not they are read back. A
second-order jet carries :math:`w^2` Hessian entries. Register pressure and compile time grow with the width.

That makes jets a good fit for a *local* derivative of statically known arity. For example, a mass-spring system is made
up of many springs, each with two ``vec3`` endpoints, so collecting the local derivatives of an elastic spring energy
requires a width of 6. Similarly, contact forces between triangle meshes are often expressed as edge-edge or
point-triangle constraints, each of which involves 4 ``vec3`` vertex positions.

Jets also generally outperform reverse mode when a function has *more outputs than inputs*, i.e. a tall Jacobian
:math:`r : \mathbb{R}^k \to \mathbb{R}^m` with :math:`m > k`. Reverse mode needs one backward sweep per output, while
one width-``k`` jet carries all ``k`` directions through a single forward pass, and any work shared by the outputs runs
once (see :ref:`the Jacobian benchmark below <jets_jacobian_performance>`).

They are the wrong tool for differentiating with respect to a problem-sized or runtime number of variables. A width of
:math:`n` cannot specialize a type, and even where it could, forward mode would cost :math:`n` directions against
reverse mode's single backward pass. Use :class:`wp.Tape <Tape>` or :func:`wp.grad <grad>` for the outer derivative, and
jets for the fixed-arity term inside it.

.. list-table::
    :header-rows: 1

    * - Task
      - Suggested tool
    * - Global gradient
      - :class:`wp.Tape <Tape>`
    * - Local gradient
      - :func:`wp.grad <grad>` or :class:`wp.Tape <Tape>`. For small in-kernel gradients,
        :func:`wp.JetSpace <JetSpace>` can sometimes be faster than :func:`wp.grad <grad>`.
    * - Local Jacobian
      - :func:`wp.JetSpace <JetSpace>`
    * - Vector-Jacobian product
      - :func:`wp.grad <grad>` or :class:`wp.Tape <Tape>`
    * - Jacobian-vector product
      - :func:`wp.JetSpace <JetSpace>` with ``directional()``
    * - Local Hessian
      - :func:`wp.JetSpace2 <JetSpace2>` if compile time allows, otherwise :func:`wp.grad <grad>` over
        :func:`wp.JetSpace <JetSpace>`
    * - Hessian-vector product
      - :func:`wp.grad <grad>` over :func:`wp.JetSpace <JetSpace>`
    * - Global sparse Jacobian or Hessian
      - If the contributions are local, :func:`wp.JetSpace <JetSpace>` or :func:`wp.JetSpace2 <JetSpace2>` for each
        local block, scattered into the global matrix

Performance
-----------

The numbers below come from the benchmarks in ``warp/examples/benchmarks``, run on an NVIDIA RTX PRO 5000 Blackwell
(laptop) GPU in float32 with about one million elements per run. Each time is the best of 20 launches for the gradient
and of 10 for the Hessian, with launch overhead included and no CUDA graph. The tetrahedron Hessian row uses 200,000
elements. Timings on this laptop GPU vary by tens of percent between runs, so read the table for ratios and orders of
magnitude. Each benchmark also prints how far the strategies' results are from each other, and the Hessian benchmark
how far they are from finite differences; these errors are reported, not enforced. The timings show the *relative* cost
of the strategies, not absolute figures to expect on other hardware or for other energies.

**Gradient** (``benchmark_jet_gradient_mesh.py``). Evaluating the energy as a width-``k`` jet is about as fast as
reverse mode, and ``wp.grad`` in the same kernel is about as fast again. First-order jets are not a speed-up over reverse
mode here; their value is that the same code also yields second derivatives. The forward-mode and in-kernel
``wp.grad`` times include evaluating the energy, while the tape time covers only its backward pass.

.. list-table::
    :header-rows: 1

    * - Element
      - ``k``
      - Forward-mode ``wp.JetSpace`` (ns/element)
      - Reverse mode, ``Tape`` (ns/element)
      - ``wp.grad`` in kernel (ns/element)
    * - Spring
      - 6
      - **0.050**
      - 0.067
      - 0.051
    * - Triangle
      - 9
      - 0.073
      - 0.088
      - **0.072**
    * - Tetrahedron
      - 12
      - 0.113
      - 0.125
      - **0.111**

**Hessian** (``benchmark_jet_hessian_mesh.py``). The in-kernel ``wp.grad`` route and the second-order jet are 2 to 9
times faster than running a tape, because the per-element Hessian stays in registers and needs a single launch.

.. list-table::
    :header-rows: 1

    * - Element
      - ``k``
      - :func:`wp.JetSpace2 <JetSpace2>` (ns/element)
      - :func:`wp.grad <grad>` over :func:`wp.JetSpace <JetSpace>` (ns/element)
      - :class:`wp.Tape <Tape>` over :func:`wp.JetSpace <JetSpace>` (ns/element)
    * - Spring
      - 6
      - **0.54**
      - 0.56
      - 4.81
    * - Triangle
      - 9
      - **1.13**
      - 2.18
      - 10.9
    * - Tetrahedron
      - 12
      - **2.28**
      - 5.41
      - 12.5

The second-order jet is fastest at run time for these small energies, but it pays for that at compile time, because its
state is :math:`O(k^2)` and NVCC compile time is super-linear in ``k``. On an RTX PRO 6000 Blackwell (a MIG 1g.24gb
slice) with CUDA 12.9, the tetrahedron module of ``benchmark_jet_hessian_mesh.py``, which holds the second-order kernel
alongside the first-order routes, took about 8.5 minutes to compile on a cold cache. When compile time matters, in-kernel ``wp.grad`` over a first-order jet
gives the same Hessian with state that grows linearly in ``k``.

.. _jets_jacobian_performance:

**Jacobian of a many-output function** (``benchmark_jet_jacobian.py``). Take a forward model with ``k = 3`` parameters
that integrates a damped pendulum for 16 steps and then reads out ``m`` values of the final state, at 200,000 points. The
integration is shared by all ``m`` readouts. The jet runs it once on a width-3 jet and evaluates every readout on that
jet. The in-kernel ``wp.grad`` route calls ``wp.grad(residual)(x, j)`` once per output, which re-runs the integration for
every row. The tape route is the per-row ``tape.backward`` loop described in :doc:`differentiability`, timed
on its backward sweeps alone.

.. list-table::
    :header-rows: 1

    * - Outputs ``m``
      - Width-3 jet (ns/element)
      - ``wp.grad`` in kernel (ns/element)
      - ``m`` tape sweeps (ns/element)
    * - 4
      - **0.13**
      - 0.38
      - 2.06
    * - 16
      - **0.19**
      - 1.39
      - 30.3
    * - 32
      - **0.78**
      - 2.81
      - 129
    * - 64
      - **1.04**
      - 5.71
      - not run
    * - 128
      - **2.08**
      - 11.3
      - not run

The jet is about 3 to 7 times faster than in-kernel ``wp.grad`` for ``m >= 4``. It is 16 times faster than the tape
at ``m = 4`` and about 160 times faster at ``m = 16`` and ``32``, because the tape launches ``m`` backward passes and
rewrites an :math:`n \times m` seed each time. The tape was not run beyond ``m = 32`` because its seeds need memory
proportional to :math:`n m^2`. Small ``m`` times are close to the launch floor.
The advantage over ``wp.grad`` comes from the shared integration and shrinks as the readouts become the larger share of
the work, because each readout costs ``k + 1`` times more on a jet. It also shrinks as ``k`` grows, since the jet's cost
scales with ``k`` and reverse mode's does not.

From ``m = 32`` on, the jet is bound by writing the Jacobian itself: at ``m = 64`` and ``128`` it stores about 740 GB/s,
near this GPU's 896 GB/s peak. Up to ``m = 16`` the Jacobian (38 MB at ``m = 16``) fits in the GPU's 64 MiB L2 cache
across repeated launches, which flatters those jet times. That is why the jet's cost per point quadruples between
``m = 16`` and ``m = 32`` and its lead over ``wp.grad`` dips there.

Store the Jacobian output-major, ``jac[j * n + i]``, so that neighboring threads write neighboring elements. Storing it
point-major, ``jac[i * m + j]``, made the jet about five times slower at ``m = 64`` (1.32 ms against 0.25 ms) and left
the ``wp.grad`` route nearly unchanged, because the jet is fast enough to be bound by those strided stores.

.. _jets_custom_derivatives:

Custom Derivatives
------------------

Some functions are only defined for floating-point inputs (e.g.,
:func:`wp.mesh_query_point() <warp._src.lang.mesh_query_point>`), and others are expensive to differentiate step by
step. In such cases, we can replace the derivative of a function with a custom implementation.

Example: Ellipsoid Closest Point
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The closest point :math:`p` on an ellipsoid with semi-axes :math:`a` to a query point :math:`q` outside it
has no closed form. It is :math:`p_i = a_i^2 q_i / (a_i^2 + t)`, where the Lagrange multiplier :math:`t` is the root of

.. math::

    f(t, q) = \sum_i \left( \frac{a_i q_i}{a_i^2 + t} \right)^2 - 1,

which a few Newton steps find. Written generically, the solver accepts jets, and calling it on a seeded query
differentiates through every step:

.. testcode::

    Jq = wp.JetSpace((wp.vec3,))  # width 3

    @wp.func
    def ellipsoid_newton_step(query: Any, axes: wp.vec3, t: Any):
        d = wp.cw_mul(axes, axes) + t * wp.vec3(1.0)
        u = wp.cw_div(wp.cw_mul(axes, query), d)
        f = wp.dot(u, u) - 1.0
        df_dt = -2.0 * wp.dot(u, wp.cw_div(u, d))
        return t - f / df_dt

    @wp.func
    def ellipsoid_point(query: Any, axes: wp.vec3, t: Any):
        axes2 = wp.cw_mul(axes, axes)
        return wp.cw_div(wp.cw_mul(axes2, query), axes2 + t * wp.vec3(1.0))

    @wp.kernel
    def ellipsoid_point_jacobian_kernel(
        query: wp.array[wp.vec3],
        axes: wp.vec3,
        jacobian: wp.array[Jq.grad_type_vec3],
    ):
        tid = wp.tid()
        jet_query = Jq.seed(query[tid])
        jet_t = Jq.constant(0.0)
        for _ in range(20):
            jet_t = ellipsoid_newton_step(jet_query, axes, jet_t)  # every step carries the derivatives
        jacobian[tid] = ellipsoid_point(jet_query, axes, jet_t).grad

    axes = wp.vec3(1.0, 0.5, 0.25)
    query = wp.array([[2.0, 1.0, 0.5]], dtype=wp.vec3)
    jacobian = wp.zeros(1, dtype=Jq.grad_type_vec3)

    wp.launch(ellipsoid_point_jacobian_kernel, dim=1, inputs=[query, axes], outputs=[jacobian])

    print(f"jacobian =\n{np.round(jacobian.numpy()[0], 4)}")

.. testoutput::

    jacobian =
    [[ 0.0921 -0.1095 -0.0182]
     [-0.1095  0.1459 -0.0054]
     [-0.0182 -0.0054  0.0505]]

Every Newton step pays for the derivatives, although only the solution's derivative is needed. Instead, run the solver
on plain values and differentiate only its solution. Because :math:`f(t, q) = 0` there, the implicit function theorem
gives the derivative of the root as

.. math::

    \frac{\partial t}{\partial q} = -\left( \frac{\partial f}{\partial t} \right)^{-1} \frac{\partial f}{\partial q}.

One more Newton step, taken on the jet query from the converged plain :math:`t`, computes exactly this: its value stays
:math:`t`, and since :math:`f` vanishes there, its gradients are :math:`-(\partial f / \partial t)^{-1}` times those
of :math:`f`. The same generic functions serve both passes:

.. testcode::

    @wp.kernel
    def ellipsoid_point_implicit_jacobian_kernel(
        query: wp.array[wp.vec3],
        axes: wp.vec3,
        jacobian: wp.array[Jq.grad_type_vec3],
    ):
        tid = wp.tid()
        t = float(0.0)
        for _ in range(20):
            t = ellipsoid_newton_step(query[tid], axes, t)  # plain floats
        jet_query = Jq.seed(query[tid])
        jet_t = ellipsoid_newton_step(jet_query, axes, t)  # one jet step: implicit function theorem
        jacobian[tid] = ellipsoid_point(jet_query, axes, jet_t).grad

    wp.launch(ellipsoid_point_implicit_jacobian_kernel, dim=1, inputs=[query, axes], outputs=[jacobian])

    print(f"jacobian =\n{np.round(jacobian.numpy()[0], 4)}")

.. testoutput::

    jacobian =
    [[ 0.0921 -0.1095 -0.0182]
     [-0.1095  0.1459 -0.0054]
     [-0.0182 -0.0054  0.0505]]

Limitations
-----------

* The width is fixed when the types are created and cannot change at runtime.
* Creating a jet space registers its arithmetic on Warp's builtins for the lifetime of the process. The effect is
  additive and happens once per width and scalar type.
* Comparisons such as ``a < b`` are not defined on jets. Write ``wp.value(a) < wp.value(b)`` instead:
  :func:`wp.value() <warp._src.lang.value>` returns plain floats unchanged, so a generic function can branch on floats and jets alike.
* Only builtins with jet overloads accept jets. Others, such as spatial queries like
  :func:`wp.mesh_query_point() <warp._src.lang.mesh_query_point>`, operate on floats only and need a custom derivative
  (see :ref:`jets_custom_derivatives`).
* Vector and matrix constructors such as ``wp.vec3(a, b, c)`` and ``wp.mat33(...)`` do not accept jets. Build vector
  and matrix jets with ``J.make_vec3()`` and ``J.make_mat33()`` (from columns), or assign components of an existing jet
  (``v[i] = s``, ``F[i, j] = s``).
* Second-order jets take scalar, ``vec2`` and ``vec3`` slots. Matrix slots are not supported; a matrix input
  decomposes into its components.
* Jet dtypes must be floating point.
* A plain scalar, vector or matrix of the jet's dtype can be either operand of a jet operation, and is treated as a
  constant. The exception is ``wp.where``, whose two branches must both be jets; wrap a plain branch with
  ``J.constant``.
* Forward-over-reverse is not supported: :func:`wp.grad <grad>` produces a native function rather than a
  :func:`@wp.func <func>`, so it cannot be called with jets.
