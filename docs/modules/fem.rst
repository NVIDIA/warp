warp.fem
=====================

.. currentmodule:: warp.fem

..
   .. toctree::
   :maxdepth: 2

The ``warp.fem`` module is designed to facilitate solving physical systems described as differential 
equations. For example, it can solve PDEs for diffusion, convection, fluid flow, and elasticity problems 
using finite element-based (FEM) Galerkin methods, and allows users to quickly experiment with various FEM
formulations and discretization schemes.

Integrands
----------

The core functionality of the FEM toolkit is the ability to integrate constant, linear and bilinear forms 
over various domains and using arbitrary interpolation basis.

The main mechanism is the ``warp.fem.integrand`` decorator, for instance: ::

      @integrand
      def linear_form(
         s: Sample,
         v: Field,
      ):
         return v(s)


      @integrand
      def diffusion_form(s: Sample, u: Field, v: Field, nu: float):
         return nu * wp.dot(
            grad(u, s),
            grad(v, s),
         )

Integrands are normal warp kernels, meaning any usual warp function can be used. 
However, they accept a few special parameters:

  - :class:`.types.Sample` contains information about the current integration sample point, such as the element index and coordinates in element.
  - :class:`.types.Field` designates an abstract field, which will be replaced at call time by the actual field type. This can be a :class:`.field.DiscreteField`, :class:`.field.TestField`, :class:`.field.TrialField` defined over arbitrary :class:`.space.FunctionSpace`.
    Fields can be evaluated at a given sample using the :func:`.operator.inner` operator, or as a shortcut using the usual call operator. 
    Several other operators are available, such as :func:`operator.grad`; see the :ref:`Operators` section.
  - :class:`.types.Domain` designates an abstract integration domain. Several operators are also provided for domains, for example in the example below evaluating the normal at the current sample position: ::
    
            @integrand
            def y_mass_form(
                s: Sample,
                domain: Domain,
                u: Field,
                v: Field,
            ):
                # Non-zero mass on vertical edges only
                nor = normal(domain, s)
                return u(s) * v(s) * nor[0]

Integrands cannot be used directly with :func:`warp.launch`, but must be called through :func:`.integrate.integrate` or :func:`.integrate.interpolate` instead.
The root integrand (``integrand`` argument to ``integrate`` or ``interpolate``) will automatically get passed :class:`.types.Sample` and :class:`.types.Domain` parameters. 
:class:`.types.Field` parameters must be passed as a dictionary in the ``fields`` argument of the launcher function, and all other standard Warp types arguments must be 
passed as a dictionary in the ``values``  argument of the launcher function, for instance: ::
    
    integrate(diffusion_form, fields={"u": trial, "v": test}, values={"nu": viscosity})


Basic workflow
--------------

The typical steps for solving a linear PDE are as follow:

 - Define a :class:`.geometry.Geometry` (grid, mesh, etc). At the moment, 2D and 3D regular grids, triangle and tetrahedron meshes are supported.
 - Define one or more :class:`.space.FunctionSpace`, by equipping the geometry elements with shape functions. See :func:`.space.make_polynomial_space`. At the moment, Lagrange polynomial shape functions up to order 3 are supported.
 - Define an integration :class:`.domain.GeometryDomain`: the geometry's cells or boundary sides, for instance
 - Integrate linear forms to build the system's right-hand-side. Define a test function over the function space using :func:`.field.make_test`,
   a :class:`.quadrature.Quadrature` formula (or let the module choose one based on the function space degree), and call :func:`.integrate.integrate` with the linear form integrand.
   The result is a :class:`warp.array` containing the integration result for each of the function space degrees of freedom.
 - Integrate bilinear forms to build the system's left-hand-side. Define a trial function over the function space using :func:`.field.make_trial`,
   then call :func:`.integrate.integrate` with the bilinear form integrand.
   The result is a :class:`warp.sparse.BsrMatrix` containing the integration result for each pair of test and trial function space degrees of freedom.
   Note that the trial and test functions do not have to be defined over the same function space, so that Mixed FEM is supported.
 - Solve the resulting linear system using the solver of your choice


The following excerpt from the introductory example ``examples/fem/example_diffusion.py`` outlines this procedure: ::

      # Grid geometry
      geo = Grid2D(n=50, cell_size=2)

      # Domain and function spaces
      domain = Cells(geometry=geo)
      scalar_space = make_polynomial_space(geo, degree=3)

      # Right-hand-side (forcing term)
      test = make_test(space=scalar_space, domain=domain)
      rhs = integrate(linear_form, fields={"v": test})

      # Weakly-imposed boundary conditions on Y sides
      boundary = BoundarySides(geo)
      bd_test = make_test(space=scalar_space, domain=boundary)
      bd_trial = make_trial(space=scalar_space, domain=boundary)
      bd_matrix = integrate(y_mass_form, fields={"u": bd_trial, "v": bd_test})

      # Diffusion form
      trial = make_trial(space=scalar_space, domain=domain)
      matrix = integrate(diffusion_form, fields={"u": trial, "v": test}, values={"nu": viscosity})

      # Assemble linear system (add diffusion and boundary condition matrices)
      bsr_axpy(x=bd_matrix, y=matrix, alpha=boundary_strength, beta=1)

      # Solve linear system using Conjugate Gradient
      x = wp.zeros_like(rhs)
      bsr_cg(matrix, b=rhs, x=x)


.. note::
   The :func:`.integrate.integrate` function does not check that the passed integrands are actually linear or bilinear forms; it is up to the user to ensure that they are.
   To solve non-linear PDEs, one can use an iterative procedure and pass the current value of the studied function :class:`DiscreteField` argument to the integrand, on which 
   arbitrary operations are permitted. However, the result of the form must remain linear in the test and trial fields.

Introductory examples
---------------------

``warp.fem`` ships with a list of examples in the ``examples/fem`` directory illustrating common model problems.

 - ``example_diffusion.py``: 2D diffusion with homogenous Neumann and Dirichlet boundary conditions
     * ``example_diffusion_3d.py``: 3D variant of the diffusion problem
 - ``example_convection_diffusion.py``: 2D convection-diffusion using semi-Lagrangian advection
     * ``example_diffusion_dg0.py``: 2D convection-diffusion using finite-volume and upwind transport
     * ``example_diffusion_dg.py``: 2D convection-diffusion using Discontinuous Galerkin aith upwind transport and Symmetric Interior Penalty
 - ``example_stokes.py``: 2D incompressible Stokes flow using mixed P_k/P_{k-1} elements
 - ``example_navier_stokes.py``: 2D Navier-Stokes flow using mixed P_k/P_{k-1} elements
 - ``example_mixed_elasticity.py``: 2D linear elasticity using mixed continuous/discontinuous P_k/P_{k-1}d elements


Advanced usages
---------------

Particle-based quadrature
^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.quadrature.PicQuadrature` provides a way to define Particle-In-Cell quadratures from a set or arbitrary particles,
which can be helpful to develop MPM-like methods.
The particles are automatically bucketed to the geometry cells when the quadrature is initialized.
This is illustrated by the ``example_stokes_transfer[_3d].py`` and ``example_apic_fluid.py`` examples.

Partitioning
^^^^^^^^^^^^

The FEM toolkit makes it possible to perform integration on a subset of the domain elements, 
possibly re-indexing degrees of freedom so that the linear system contains the local ones only.
This is useful for distributed computation (see ``examples/fem/example_diffusion_mgpu.py``), or simply to limit the simulation domain to a subset of active cells (see ``examples/fem/example_stokes_transfer.py``).

A partition of the simulation geometry can be defined using subclasses of :class:`.geometry.GeometryPartition`
such as :class:`.geometry.LinearGeometryPartition`  or :class:`.geometry.ExplicitGeometryPartition`.

Function spaces can then be partitioned according to the geometry partition using :func:`.space.make_space_partition`. 
The resulting :class:`.space.SpacePartition` object allows translating between space-wide and partition-wide node indices, 
and differentiating interior, frontier and exterior nodes.

Memory management
^^^^^^^^^^^^^^^^^

Several ``warp.fem`` functions require allocating temporary buffers to perform their computations. 
If such functions are called many times in a tight loop, those many allocations and de-allocations may degrade performance.
To overcome this issue, a :class:`.cache.TemporaryStore` object may be created to persist and re-use temporary allocations across calls,
either globally using :func:`set_default_temporary_store` or at a per-function granularity using the corresponding argument.

.. _Operators:

Operators
---------
.. autofunction:: warp.fem.operator.position(domain: Domain, s: Sample)
.. autofunction:: warp.fem.operator.normal(domain: Domain, s: Sample)
.. autofunction:: warp.fem.operator.lookup(domain: Domain, x)
.. autofunction:: warp.fem.operator.measure(domain: Domain, s: Sample)
.. autofunction:: warp.fem.operator.measure_ratio(domain: Domain, s: Sample)

.. autofunction:: warp.fem.operator.degree(f: Field)
.. autofunction:: warp.fem.operator.inner(f: Field, s: Sample)
.. autofunction:: warp.fem.operator.outer(f: Field, s: Sample)
.. autofunction:: warp.fem.operator.grad(f: Field, s: Sample)
.. autofunction:: warp.fem.operator.grad_outer(f: Field, s: Sample)
.. autofunction:: warp.fem.operator.at_node(f: Field, s: Sample)

.. autofunction:: warp.fem.operator.D(f: Field, s: Sample)
.. autofunction:: warp.fem.operator.div(f: Field, s: Sample)
.. autofunction:: warp.fem.operator.jump(f: Field, s: Sample)
.. autofunction:: warp.fem.operator.average(f: Field, s: Sample)
.. autofunction:: warp.fem.operator.grad_jump(f: Field, s: Sample)
.. autofunction:: warp.fem.operator.grad_average(f: Field, s: Sample)

.. autofunction:: warp.fem.operator.operator

Integration
-----------

.. autofunction:: warp.fem.integrate.integrate
.. autofunction:: warp.fem.integrate.interpolate

.. autofunction:: warp.fem.operator.integrand

.. class:: warp.fem.types.Sample

   Per-sample point context for evaluating fields and related operators in integrands.


Geometry
--------

.. autoclass:: warp.fem.geometry.Geometry
   :members:

.. autoclass:: warp.fem.geometry.GeometryPartition
   :members:

.. autoclass:: warp.fem.domain.GeometryDomain
   :members:

.. autoclass:: warp.fem.quadrature.Quadrature
   :members:

.. autoclass:: warp.fem.geometry.LinearGeometryPartition
   :members:

.. autoclass:: warp.fem.geometry.ExplicitGeometryPartition
   :members:

.. autoclass:: warp.fem.quadrature.PicQuadrature
   :members:

Function Spaces
---------------

.. autofunction:: warp.fem.space.make_polynomial_space

.. autofunction:: warp.fem.space.make_space_partition

.. autofunction:: warp.fem.space.make_space_restriction

.. autoclass:: warp.fem.space.FunctionSpace
   :members:

.. autoclass:: warp.fem.space.SpacePartition
   :members:

.. autoclass:: warp.fem.space.SpaceRestriction
   :members:

.. autoclass:: warp.fem.space.DofMapper
   :members:

.. autoclass:: warp.fem.space.SymmetricTensorMapper
   :members:

Fields
------

.. autofunction:: warp.fem.field.make_test

.. autofunction:: warp.fem.field.make_trial

.. autofunction:: warp.fem.field.make_restriction

.. autoclass:: warp.fem.field.DiscreteField
   :members:

.. autoclass:: warp.fem.field.FieldRestriction
   :members:

.. autoclass:: warp.fem.field.TestField
   :members:

.. autoclass:: warp.fem.field.TrialField
   :members:


Memory management
-----------------

.. autoclass:: warp.fem.cache.TemporaryStore
   :members:

.. autofunction:: warp.fem.cache.borrow_temporary

.. autoclass:: warp.fem.cache.Temporary
   :members:

.. autofunction:: warp.fem.cache.borrow_temporary_like

.. autofunction:: warp.fem.cache.set_default_temporary_store
