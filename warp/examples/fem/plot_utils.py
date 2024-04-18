from typing import Set

import numpy as np

from warp.fem import DiscreteField


def plot_grid_surface(field, axes=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if axes is None:
        fig, axes = plt.subplots(subplot_kw={"projection": "3d"})

    node_positions = field.space.node_grid()

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

    node_positions = field.space.node_positions().numpy()

    triangulation = Triangulation(
        x=node_positions[:, 0], y=node_positions[:, 1], triangles=field.space.node_triangulation()
    )

    Z = field.dof_values.numpy()

    # Plot the surface.
    return axes.plot_trisurf(triangulation, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)


def plot_scatter_surface(field, axes=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if axes is None:
        fig, axes = plt.subplots(subplot_kw={"projection": "3d"})

    X, Y = field.space.node_positions().numpy().T

    # Make data.
    Z = field.dof_values.numpy().reshape(X.shape)

    # Plot the surface.
    return axes.scatter(X, Y, Z, c=Z, cmap=cm.coolwarm)


def plot_surface(field, axes=None):
    if hasattr(field.space, "node_grid"):
        return plot_grid_surface(field, axes)
    elif hasattr(field.space, "node_triangulation"):
        return plot_tri_surface(field, axes)
    else:
        return plot_scatter_surface(field, axes)


def plot_grid_color(field, axes=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if axes is None:
        fig, axes = plt.subplots()

    node_positions = field.space.node_grid()

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

    node_positions = field.space.node_positions().numpy()

    # Make data.
    X = node_positions[:, 0]
    Y = node_positions[:, 1]

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

    node_positions = field.space.node_grid()

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

    X, Y, Z = field.space.node_positions().numpy().T

    # Make data.
    f = field.dof_values.numpy().reshape(X.shape)

    # Plot the surface.
    return axes.scatter(X, Y, Z, c=f, cmap=cm.coolwarm)


def plot_3d_velocities(field, axes=None):
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots(subplot_kw={"projection": "3d"})

    X, Y, Z = field.space.node_positions().numpy().T

    vel = field.dof_values.numpy()
    u = np.ascontiguousarray(vel[:, 0])
    v = np.ascontiguousarray(vel[:, 1])
    w = np.ascontiguousarray(vel[:, 2])

    u = u.reshape(X.shape)
    v = v.reshape(X.shape)
    w = w.reshape(X.shape)

    return axes.quiver(X, Y, Z, u, v, w, length=1.0 / X.shape[0], normalize=False)


class Plot:
    def __init__(self, stage=None, default_point_radius=0.01):
        self.default_point_radius = default_point_radius

        self._surfaces = {}
        self._surface_vectors = {}
        self._volumes = {}

        self._usd_renderer = None
        if stage is not None:
            try:
                from warp.render import UsdRenderer

                self._usd_renderer = UsdRenderer(stage)
            except Exception as err:
                print(f"Could not initialize UsdRenderer for stage '{stage}': {err}.")

    def begin_frame(self, time):
        if self._usd_renderer is not None:
            self._usd_renderer.begin_frame(time=time)

    def end_frame(self):
        if self._usd_renderer is not None:
            self._usd_renderer.end_frame()

    def add_surface(self, name: str, field: DiscreteField):
        if self._usd_renderer is not None:
            points_2d = field.space.node_positions().numpy()
            values = field.dof_values.numpy()
            points_3d = np.hstack((points_2d, values.reshape(-1, 1)))

            if hasattr(field.space, "node_triangulation"):
                indices = field.space.node_triangulation()
                self._usd_renderer.render_mesh(name, points=points_3d, indices=indices)
            else:
                self._usd_renderer.render_points(name, points=points_3d, radius=self.default_point_radius)

        if name not in self._surfaces:
            field_clone = field.space.make_field(space_partition=field.space_partition)
            self._surfaces[name] = (field_clone, [])

        self._surfaces[name][1].append(field.dof_values.numpy())

    def add_surface_vector(self, name: str, field: DiscreteField):
        if self._usd_renderer is not None:
            points_2d = field.space.node_positions().numpy()
            values = field.dof_values.numpy()
            points_3d = np.hstack((points_2d + values, np.zeros_like(points_2d[:, 0]).reshape(-1, 1)))

            if hasattr(field.space, "node_triangulation"):
                indices = field.space.node_triangulation()
                self._usd_renderer.render_mesh(name, points=points_3d, indices=indices)
            else:
                self._usd_renderer.render_points(name, points=points_3d, radius=self.default_point_radius)

        if name not in self._surface_vectors:
            field_clone = field.space.make_field(space_partition=field.space_partition)
            self._surface_vectors[name] = (field_clone, [])

        self._surface_vectors[name][1].append(field.dof_values.numpy())

    def add_volume(self, name: str, field: DiscreteField):
        if self._usd_renderer is not None:
            points_3d = field.space.node_positions().numpy()
            values = field.dof_values.numpy()

            self._usd_renderer.render_points(name, points_3d, radius=values)

        if name not in self._volumes:
            field_clone = field.space.make_field(space_partition=field.space_partition)
            self._volumes[name] = (field_clone, [])

        self._volumes[name][1].append(field.dof_values.numpy())

    def plot(self, streamlines: Set[str] = None):
        if streamlines is None:
            streamlines = []
        return self._plot_matplotlib(streamlines)

    def _plot_matplotlib(self, streamlines: Set[str] = None):
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        if streamlines is None:
            streamlines = []

        def make_animation(ax, field, values, plot_func, num_frames: int):
            def animate(i):
                ax.clear()
                field.dof_values = values[i]
                return plot_func(field, axes=ax)

            return animation.FuncAnimation(
                ax.figure,
                animate,
                interval=30,
                blit=False,
                frames=len(values),
            )

        for _name, (field, values) in self._surfaces.items():
            field.dof_values = values[0]
            ax = plot_surface(field).axes

            if len(values) > 1:
                _anim = make_animation(ax, field, values, plot_func=plot_surface, num_frames=len(values))

        for name, (field, values) in self._surface_vectors.items():
            field.dof_values = values[0]
            if name in streamlines and hasattr(field.space, "node_grid"):
                ax = plot_grid_streamlines(field).axes
            else:
                ax = plot_velocities(field).axes

                if len(values) > 1:
                    _anim = make_animation(ax, field, values, plot_func=plot_velocities, num_frames=len(values))

        for _name, (field, values) in self._volumes.items():
            field.dof_values = values[0]
            ax = plot_3d_scatter(field).axes

        plt.show()
