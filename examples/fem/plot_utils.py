import numpy as np


def plot_grid_surface(field, axes=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if axes is None:
        fig, axes = plt.subplots(subplot_kw={"projection": "3d"})

    node_positions = field.space.node_positions()

    # Make data.
    X = node_positions[0]
    Y = node_positions[1]
    Z = field.dof_values.numpy().reshape(X.shape)

    # Plot the surface.
    return axes.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)


def plot_tri_surface(field, axes=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.tri.triangulation import Triangulation

    if axes is None:
        fig, axes = plt.subplots(subplot_kw={"projection": "3d"})

    node_positions = field.space.node_positions()

    triangulation = Triangulation(
        x=node_positions[0], y=node_positions[1], triangles=field.space.node_triangulation()
    )

    Z = field.dof_values.numpy()

    # Plot the surface.
    return axes.plot_trisurf(triangulation, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)


def plot_scatter_surface(field, axes=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if axes is None:
        fig, axes = plt.subplots(subplot_kw={"projection": "3d"})

    X, Y = field.space.node_positions()

    # Make data.
    Z = field.dof_values.numpy().reshape(X.shape)

    # Plot the surface.
    return axes.scatter(X, Y, Z, c=Z, cmap=cm.coolwarm)


def plot_surface(field, axes=None):
    if hasattr(field.space, "node_triangulation"):
        return plot_tri_surface(field, axes)
    else:
        try:
            return plot_grid_surface(field, axes)
        except:
            return plot_scatter_surface(field, axes)


def plot_grid_color(field, axes=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if axes is None:
        fig, axes = plt.subplots()

    node_positions = field.space.node_positions()

    # Make data.
    X = node_positions[0]
    Y = node_positions[1]
    Z = field.dof_values.numpy().reshape(X.shape)

    # Plot the surface.
    return axes.pcolormesh(X, Y, Z, cmap=cm.coolwarm)


def plot_velocities(field, axes=None):
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots()

    node_positions = field.space.node_positions()

    # Make data.
    X = node_positions[0]
    Y = node_positions[1]

    vel = field.dof_values.numpy()
    u = np.ascontiguousarray(vel[:, 0])
    v = np.ascontiguousarray(vel[:, 1])

    u = u.reshape(X.shape)
    v = v.reshape(X.shape)

    return axes.quiver(X, Y, u, v)


def plot_grid_streamlines(field, axes=None):
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots()

    node_positions = field.space.node_positions()

    # Make data.
    X = node_positions[0][:, 0]
    Y = node_positions[1][0, :]

    vel = field.dof_values.numpy()
    u = np.ascontiguousarray(vel[:, 0])
    v = np.ascontiguousarray(vel[:, 1])

    u = np.transpose(u.reshape(node_positions[0].shape))
    v = np.transpose(v.reshape(node_positions[0].shape))

    splot = axes.streamplot(X, Y, u, v, density=2)
    splot.axes = axes
    return splot


def plot_3d_scatter(field, axes=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if axes is None:
        fig, axes = plt.subplots(subplot_kw={"projection": "3d"})

    X, Y, Z = field.space.node_positions()

    # Make data.
    f = field.dof_values.numpy().reshape(X.shape)

    # Plot the surface.
    return axes.scatter(X, Y, Z, c=f, cmap=cm.coolwarm)


def plot_3d_velocities(field, axes=None):
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots(subplot_kw={"projection": "3d"})

    X, Y, Z = field.space.node_positions()

    vel = field.dof_values.numpy()
    u = np.ascontiguousarray(vel[:, 0])
    v = np.ascontiguousarray(vel[:, 1])
    w = np.ascontiguousarray(vel[:, 2])

    u = u.reshape(X.shape)
    v = v.reshape(X.shape)
    w = w.reshape(X.shape)

    return axes.quiver(X, Y, Z, u, v, w, length=1.0 / X.shape[0], normalize=False)
