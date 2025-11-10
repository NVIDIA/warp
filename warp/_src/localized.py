import ast
import random

# ==============================================================================
# Layout Class
# ==============================================================================


class Layout:
    """A layout descriptor for kernel partitioning, similar to CuTe's Layout concept.

    A Layout maps logical multi-dimensional coordinates to linear offsets using
    shape (extent) and stride (step size) in each dimension.

    Examples:
        # 2D round-robin partition: 5x5 blocks with strides (2, 20)
        layout = wp.Layout(shape=(5, 5), stride=(2, 20))

        # 2D quadrant partition: 5x5 blocks with strides (1, 10)
        layout = wp.Layout(shape=(5, 5), stride=(1, 10))

        # Row-major 1D partition
        layout = wp.Layout(shape=(10,), stride=(1,))

        # Create from string (backward compatibility)
        layout = wp.Layout.from_string("(5,5):(2,20)")

    Attributes:
        shape (tuple[int, ...]): Extent in each dimension (number of blocks per dimension)
        stride (tuple[int, ...]): Step size in each dimension
        rank (int): Number of dimensions
        size (int): Total number of elements (product of shape)
    """

    def __init__(self, shape: tuple[int, ...] | list[int], stride: tuple[int, ...] | list[int] | None = None):
        """Initialize a Layout with shape and stride.

        Args:
            shape: Extent in each dimension
            stride: Step size in each dimension. If None, computes row-major strides.

        Raises:
            ValueError: If shape is empty or if shape and stride have different lengths
        """
        if not shape or len(shape) == 0:
            raise ValueError("Layout shape must have at least 1 dimension")

        self.shape = tuple(shape)
        self.rank = len(self.shape)

        # Compute strides if not provided (row-major by default)
        if stride is None:
            self.stride = self._compute_row_major_strides(self.shape)
        else:
            if len(stride) != len(shape):
                raise ValueError(
                    f"Shape and stride must have same length. "
                    f"Got shape={shape} (length {len(shape)}), stride={stride} (length {len(stride)})"
                )
            self.stride = tuple(stride)

        # Compute total size
        self.size = 1
        for s in self.shape:
            self.size *= s

    @staticmethod
    def _compute_row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute row-major strides from shape."""
        if len(shape) == 1:
            return (1,)

        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)

    @staticmethod
    def _compute_col_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute column-major strides from shape."""
        if len(shape) == 1:
            return (1,)

        strides = [1] * len(shape)
        for i in range(1, len(shape)):
            strides[i] = strides[i - 1] * shape[i - 1]
        return tuple(strides)

    @classmethod
    def row_major(cls, shape: tuple[int, ...] | list[int]):
        """Create a row-major layout.

        Example:
            Layout.row_major((4, 8)) -> shape=(4,8), stride=(8,1)
        """
        return cls(shape=shape, stride=None)

    @classmethod
    def col_major(cls, shape: tuple[int, ...] | list[int]):
        """Create a column-major layout.

        Example:
            Layout.col_major((4, 8)) -> shape=(4,8), stride=(1,4)
        """
        strides = cls._compute_col_major_strides(tuple(shape))
        return cls(shape=shape, stride=strides)

    @classmethod
    def from_string(cls, layout_str: str):
        """Parse a layout from CuTe-style string format '(shape):(stride)'.

        Args:
            layout_str: String in format '(a,b,c):(d,e,f)'

        Returns:
            Layout object

        Raises:
            ValueError: If string format is invalid

        Example:
            Layout.from_string("(5,5):(2,20)")
        """
        parts = layout_str.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid layout format: {layout_str}. Expected format: '(a,b,c):(d,e,f)'")

        shape_str, stride_str = parts

        # Parse shape tuple
        shape_str = shape_str.strip()
        if not (shape_str.startswith("(") and shape_str.endswith(")")):
            raise ValueError(f"Invalid shape format: {shape_str}. Expected format: '(a,b,c)'")
        shape = tuple(int(x.strip()) for x in shape_str[1:-1].split(",") if x.strip())

        # Parse stride tuple
        stride_str = stride_str.strip()
        if not (stride_str.startswith("(") and stride_str.endswith(")")):
            raise ValueError(f"Invalid stride format: {stride_str}. Expected format: '(d,e,f)'")
        stride = tuple(int(x.strip()) for x in stride_str[1:-1].split(",") if x.strip())

        return cls(shape=shape, stride=stride)

    def to_string(self) -> str:
        """Convert layout to CuTe-style string format '(shape):(stride)'.

        Returns:
            String representation

        Example:
            layout.to_string() -> "(5,5):(2,20)"
        """
        shape_str = f"({','.join(map(str, self.shape))})"
        stride_str = f"({','.join(map(str, self.stride))})"
        return f"{shape_str}:{stride_str}"

    def get_size(self) -> int:
        """Get the total number of elements (product of shape dimensions).

        Returns:
            Product of all shape dimensions

        Example:
            layout = Layout(shape=(4, 8, 2))
            layout.get_size() -> 64  # 4 * 8 * 2

        Note:
            This is equivalent to accessing the `size` attribute directly.
        """
        return self.size

    def coord_to_offset(self, *coords: int) -> int:
        """Map multi-dimensional coordinate to linear offset.

        Args:
            *coords: Coordinate in each dimension

        Returns:
            Linear offset computed as sum(coord[i] * stride[i])

        Example:
            layout = Layout(shape=(4,8), stride=(8,1))
            layout.coord_to_offset(2, 3) -> 19  # (2*8 + 3*1)
        """
        if len(coords) != self.rank:
            raise ValueError(f"Expected {self.rank} coordinates, got {len(coords)}")

        offset = 0
        for i, c in enumerate(coords):
            if c < 0 or c >= self.shape[i]:
                raise ValueError(f"Coordinate {c} out of bounds for dimension {i} with shape {self.shape[i]}")
            offset += c * self.stride[i]
        return offset

    def linear_to_coord(self, index: int) -> tuple[int, ...]:
        """Convert linear index to multi-dimensional coordinate based on shape.

        This performs the inverse of converting coordinates to a linear index
        in row-major order through the shape.

        Args:
            index: Linear index (0-based) in range [0, size)

        Returns:
            Tuple of coordinates corresponding to the linear index

        Raises:
            ValueError: If index is out of bounds

        Example:
            layout = Layout(shape=(5, 10), stride=(10, 20))
            layout.linear_to_coord(7) -> (0, 7)
            layout.linear_to_coord(12) -> (1, 2)
        """
        if index < 0 or index >= self.size:
            raise ValueError(f"Index {index} out of bounds for size {self.size}")

        coords = []
        remaining = index

        # Convert from linear index to multi-dimensional coordinate (row-major)
        for i in range(self.rank - 1, -1, -1):
            coords.append(remaining % self.shape[i])
            remaining //= self.shape[i]

        # Reverse to get correct order
        coords.reverse()
        return tuple(coords)

    def linear_to_offset(self, index: int) -> int:
        """Convert linear index to layout offset.

        This first converts the linear index to multi-dimensional coordinates
        (based on shape), then computes the layout offset using the strides.

        Args:
            index: Linear index (0-based) in range [0, size)

        Returns:
            Layout offset computed as sum(coord[i] * stride[i])

        Raises:
            ValueError: If index is out of bounds

        Example:
            layout = Layout(shape=(5, 10), stride=(10, 20))
            # Linear index 7 -> coord (0, 7) -> offset 0*10 + 7*20 = 140
            layout.linear_to_offset(7) -> 140
            # Linear index 12 -> coord (1, 2) -> offset 1*10 + 2*20 = 50
            layout.linear_to_offset(12) -> 50
        """
        coord = self.linear_to_coord(index)
        return self.coord_to_offset(*coord)

    def is_unique(self) -> bool:
        """Check if layout has no broadcast dimensions (all strides non-zero).

        Returns:
            True if no broadcast dimensions exist
        """
        return all(s != 0 for s in self.stride)

    def is_coalesced(self) -> bool:
        """Check if layout has unit stride in the innermost (last) dimension.

        Returns:
            True if innermost stride is 1
        """
        return self.stride[-1] == 1

    def __call__(self, *args) -> int:
        """Make Layout callable like CuTe's Layout function object.

        Supports three modes:
        1. Single integer: treats as linear index, returns layout offset
        2. Single tuple/list: unpacks as coordinates, returns layout offset
        3. Multiple arguments: treats as coordinates, returns layout offset

        Args:
            *args: Either a single linear index, a tuple/list of coordinates,
                   or multiple coordinate arguments

        Returns:
            Layout offset

        Examples:
            layout = Layout(shape=(5, 10), stride=(10, 20))

            # Linear index mode (single integer)
            layout(7)      # → 140 (converts 7 to coord (0,7), then 0*10+7*20)
            layout(12)     # → 50  (converts 12 to coord (1,2), then 1*10+2*20)

            # Coordinate mode (tuple/list)
            layout((0, 7)) # → 140 (0*10 + 7*20)
            layout([1, 2]) # → 50  (1*10 + 2*20)

            # Coordinate mode (multiple arguments)
            layout(0, 7)   # → 140 (0*10 + 7*20)
            layout(1, 2)   # → 50  (1*10 + 2*20)
        """
        if len(args) == 1:
            arg = args[0]
            # Check if argument is a tuple or list (coordinates)
            if isinstance(arg, (tuple, list)):
                return self.coord_to_offset(*arg)
            # Otherwise treat as linear index
            else:
                return self.linear_to_offset(arg)
        else:
            # Multiple arguments: coordinates → offset
            return self.coord_to_offset(*args)

    def __repr__(self) -> str:
        return f"Layout(shape={self.shape}, stride={self.stride})"

    def __str__(self) -> str:
        return self.to_string()

    def __eq__(self, other) -> bool:
        if not isinstance(other, Layout):
            return False
        return self.shape == other.shape and self.stride == other.stride


# ==============================================================================
# Partitioning Policies
# ==============================================================================


class PartitionDesc:
    """Result of a partitioning policy.

    Contains the partition layout and optional offset information for
    iterating over the partitioned work.

    Attributes:
        partition: Layout describing the partitioning scheme
        offsets: List of offsets for iterating over partitions (optional)
        block_shape: Shape of each block/partition (optional)
        offset_layout: Layout for offset indexing (optional)
        partition_cute: cute.Layout - hierarchical cute layout combining offset_layout and partition
        partition_inverse_cute: cute.Layout - right inverse of partition_cute
    """

    def __init__(
        self,
        partition: Layout,
        offsets: list[int] | None = None,
        block_shape: tuple[int, ...] | None = None,
        offset_layout: Layout | None = None,
    ):
        self.partition = partition
        self.block_shape = block_shape
        self.offset_layout = offset_layout

        # Compute offsets from offset_layout if provided, otherwise use explicit offsets or default
        if offset_layout is not None:
            self.offsets = [int(offset_layout(i)) for i in range(offset_layout.size)]
        elif offsets is not None:
            self.offsets = offsets
        else:
            self.offsets = list(range(partition.size))

        # Compute cute layouts if offset_layout is provided
        self.partition_cute = None
        self.partition_inverse_cute = None
        if offset_layout is not None:
            try:
                import pycute

                # Create hierarchical cute layout: ((offset_shape), (partition_shape))
                self.partition_cute = pycute.Layout(
                    (offset_layout.shape, partition.shape), (offset_layout.stride, partition.stride)
                )
                # Compute right inverse for mapping global index -> (proc_id, local_id)
                try:
                    self.partition_inverse_cute = pycute.right_inverse(self.partition_cute)
                except Exception as e:
                    print(f"Warning: Could not compute partition inverse: {e}")
                    self.partition_inverse_cute = None
            except ImportError:
                print("Warning: pycute not available, partition_cute will be None")

        # print(f"PartitionDesc: {self}")
        # print(f"  Partition: {self.partition}")
        # print(f"  Offsets: {self.offsets}")
        # print(f"  Block shape: {self.block_shape}")
        # print(f"  Offset layout: {self.offset_layout}")
        # print(f"  Partition cute: {self.partition_cute}")
        # print(f"  Partition inverse cute: {self.partition_inverse_cute}")

    def __repr__(self) -> str:
        return f"PartitionDesc(partition={self.partition}, offsets={self.offsets[:5]}{'...' if len(self.offsets) > 5 else ''}, block_shape={self.block_shape}, offset_layout={self.offset_layout})"

    def get_element_owner(self, elem_idx: int) -> int | None:
        """Compute which place owns a given element index.

        Uses the inverse partition to map from a global element index to the
        place (processor) that owns it.

        Args:
            elem_idx: Global element index

        Returns:
            Place index (int) that owns the element, or None if the element
            is not in the partition or inverse is not available
        """
        if self.partition_inverse_cute is None or self.partition_cute is None or self.offset_layout is None:
            return None

        try:
            import pycute

            # Apply inverse partition to get coordinates in partition space
            inv_idx = self.partition_inverse_cute(elem_idx)
            # Convert flat index to hierarchical coordinates
            coords = pycute.idx2crd(inv_idx, self.partition_cute.shape)
            # Extract processor coordinate (first element of hierarchical tuple)
            proc_coord = coords[0] if isinstance(coords, tuple) else coords
            # Convert processor coordinate to flat place index
            place = pycute.crd2idx(proc_coord, self.offset_layout.shape)
            return int(place)
        except Exception:
            # Element not in partition or invalid mapping
            return None


def blocked(
    dim: tuple[int, ...] | list[int] | None = None, places: int | tuple[int, ...] | list[int] | Layout | None = None
):
    """Create a blocked (contiguous) partitioning scheme or policy function.

    Can be used in two ways:

    1. Direct mode (legacy): Provide both dim and places to get a PartitionDesc
       result = wp.blocked(dim=(10, 10), places=4)

    2. Policy mode (new): Provide no arguments to get a policy function
       policy = wp.blocked()
       # Later: desc = policy(dim=(10, 10), streams=streams)

    Divides the flattened problem space into contiguous blocks. Each place gets
    a contiguous chunk of elements for good spatial locality.

    Algorithm:
    1. Compute total_size = product of dim
    2. Compute nplaces from places/streams argument
    3. part_size = total_size / nplaces (size of each chunk)
    4. Partition layout = (part_size, 1) - each place processes part_size contiguous elements
    5. Offsets layout = (nplaces, part_size) - stride by part_size to get to next block

    Args:
        dim: Total problem dimensions (e.g., (10, 10) = 100 elements). Optional.
        places: Number of partitions. Optional.
                - int: Total number of blocks
                - tuple/list: Shape to compute product (e.g., (2, 2) = 4 places)
                - Layout: Use layout.size as number of places
                - None: Will be inferred from streams when used as policy

    Returns:
        If both dim and places provided: PartitionDesc
        If neither provided: Policy function that takes (dim, streams) -> PartitionDesc

    Examples:
        # Direct mode (legacy)
        result = blocked(dim=(10, 10), places=4)

        # Policy mode (new)
        policy = blocked()
        wp.localized.launch_tiled(kernel, dim=(10, 10), mapping=policy, streams=streams)
    """
    # Policy mode: return a function
    if dim is None and places is None:

        def blocked_policy(dim, streams):
            return blocked(dim=dim, places=len(streams))

        return blocked_policy

    # Direct mode: compute PartitionDesc
    if dim is None or places is None:
        raise ValueError("blocked() requires either both dim and places, or neither (for policy mode)")

    dim = tuple(dim) if isinstance(dim, list) else dim

    # Compute total size (flatten the problem space)
    total_size = 1
    for d in dim:
        total_size *= int(d)  # Ensure int

    # Parse places argument to get number of places
    if isinstance(places, Layout):
        nplaces = int(places.size)
    elif isinstance(places, int):
        nplaces = int(places)
    else:
        # tuple/list: compute product
        places_tuple = tuple(places)
        nplaces = 1
        for p in places_tuple:
            nplaces *= int(p)

    # Compute part_size (size of each chunk) - ensure int
    part_size = int((total_size + nplaces - 1) // nplaces)  # Ceiling division

    # Partition layout: each place processes part_size contiguous elements
    partition = Layout(shape=(part_size, 1), stride=(1, 1))

    # Offsets layout: stride by part_size to get to next place
    offsets_layout = Layout(shape=(nplaces,), stride=(part_size,))

    return PartitionDesc(partition=partition, offset_layout=offsets_layout, block_shape=(part_size,))


def cyclic(
    dim: tuple[int, ...] | list[int] | None = None,
    places: int | tuple[int, ...] | list[int] | Layout | None = None,
    block_size: int | tuple[int, ...] | None = None,
):
    """Create a cyclic (round-robin) partitioning scheme or policy function.

    Can be used in two ways:

    1. Direct mode (legacy): Provide both dim and places to get a PartitionDesc
       result = wp.cyclic(dim=(12, 8), places=(2, 2))

    2. Policy mode (new): Provide no arguments to get a policy function
       policy = wp.cyclic()
       # Later: desc = policy(dim=(12, 8), streams=streams)

    Distributes work in a round-robin fashion. Each place gets every Nth element
    where N is the number of places in that dimension.

    For a 2D grid with places (2, 2):
    - Place (0,0) gets elements at (0,0), (0,2), (0,4), (2,0), (2,2), (2,4), ...
    - Place (0,1) gets elements at (0,1), (0,3), (0,5), (2,1), (2,3), (2,5), ...
    - etc.

    Args:
        dim: Total problem dimensions (e.g., (12, 8) for 12x8 grid). Optional.
        places: Grid arrangement of places. Optional.
                - int: Total number (tries to factor into grid)
                - tuple: Grid shape (e.g., (2, 2) for 2x2 grid)
                - Layout: Use layout.shape as grid
                - None: Will be inferred from streams when used as policy
        block_size: Not used (reserved for future block-cyclic)

    Returns:
        If both dim and places provided: PartitionDesc
        If neither provided: Policy function that takes (dim, streams) -> PartitionDesc

    Examples:
        # Direct mode (legacy)
        result = cyclic(dim=(12, 8), places=(2, 2))

        # Policy mode (new)
        policy = cyclic()
        wp.localized.launch_tiled(kernel, dim=(12, 8), mapping=policy, streams=streams)
    """
    # Policy mode: return a function
    if dim is None and places is None:

        def cyclic_policy(dim, streams):
            return cyclic(dim=dim, places=len(streams), block_size=block_size)

        return cyclic_policy

    # Direct mode: compute PartitionDesc
    if dim is None or places is None:
        raise ValueError("cyclic() requires either both dim and places, or neither (for policy mode)")
    dim = tuple(int(d) for d in dim) if isinstance(dim, list) else tuple(int(d) for d in dim)
    rank = len(dim)

    # Parse places
    if isinstance(places, Layout):
        places_shape = places.shape
    elif isinstance(places, int):
        if rank == 1:
            places_shape = (places,)
        elif rank == 2:
            import math

            sqrt_places = int(math.sqrt(places))
            while places % sqrt_places != 0 and sqrt_places > 1:
                sqrt_places -= 1
            places_shape = (sqrt_places, places // sqrt_places)
        else:
            places_shape = tuple([places] + [1] * (rank - 1))
    else:
        places_shape = tuple(int(p) for p in places)

    # Validate dimensions match
    if len(places_shape) != rank:
        raise ValueError(f"Places rank {len(places_shape)} doesn't match dim rank {rank}")

    # Compute partition shape: each place gets dim/places elements
    partition_shape = tuple(
        int((dim[i] + places_shape[i] - 1) // places_shape[i])  # Ceiling division
        for i in range(rank)
    )

    # Partition stride: for round-robin in linearized column-major layout
    # For column-major: element (i, j, ...) at position i + j*dim[0] + k*dim[0]*dim[1] + ...
    # Stride in dimension i: places_shape[i] × product of all dimensions before i
    partition_stride = []
    for i in range(rank):
        stride = places_shape[i]
        for j in range(i):
            stride *= dim[j]
        partition_stride.append(int(stride))
    partition_stride = tuple(partition_stride)

    partition = Layout(shape=partition_shape, stride=partition_stride)

    # Offset layout: to compute base offset for each place in dim-space
    # Offsets pattern: (places):(1, dim[0])
    if rank == 1:
        offset_stride = (1,)
    elif rank == 2:
        # Offset stride: (1, dim[0])
        # Moving to next row of places: offset by 1
        # Moving to next col of places: offset by dim[0] (row count)
        offset_stride = (1, int(dim[0]))
    else:
        # General case: row-major strides
        offset_stride = [1] * rank
        for i in range(rank - 2, -1, -1):
            offset_stride[i] = offset_stride[i + 1] * dim[i + 1]
        offset_stride = tuple(int(s) for s in offset_stride)

    offsets_layout = Layout(shape=places_shape, stride=offset_stride)

    return PartitionDesc(partition=partition, offset_layout=offsets_layout, block_shape=partition_shape)


def block_cyclic(
    dim: tuple[int, ...] | list[int],
    places: int | tuple[int, ...] | list[int] | Layout,
    block_size: int | tuple[int, ...],
) -> PartitionDesc:
    """Create a block-cyclic partitioning scheme.

    Combines blocked and cyclic approaches: distributes blocks of work
    in a round-robin fashion. Good balance between locality and load balancing.

    Args:
        dim: Total problem dimensions
        places: Number of partitions or their arrangement
        block_size: Size of blocks to distribute cyclically

    Returns:
        PartitionDesc with partition layout and offsets
    """
    dim = tuple(dim) if isinstance(dim, list) else dim
    rank = len(dim)

    # Parse places
    if isinstance(places, Layout):
        places_shape = places.shape
    elif isinstance(places, int):
        if rank == 1:
            places_shape = (places,)
        else:
            import math

            sqrt_places = int(math.sqrt(places))
            while places % sqrt_places != 0 and sqrt_places > 1:
                sqrt_places -= 1
            if rank == 2:
                places_shape = (sqrt_places, places // sqrt_places)
            else:
                places_shape = tuple([places] + [1] * (rank - 1))
    else:
        places_shape = tuple(places)

    # Parse block_size
    if isinstance(block_size, int):
        block_size_tuple = tuple([block_size] * rank)
    else:
        block_size_tuple = tuple(block_size)

    # Compute block-cyclic strides
    stride = tuple(block_size_tuple[i] * places_shape[i] for i in range(rank))

    partition = Layout(shape=places_shape, stride=stride)

    return PartitionDesc(partition=partition, block_shape=block_size_tuple)


# ==============================================================================
# Utility Functions
# ==============================================================================


def parse_cute_partition(partition_str):
    """Parse a Cute layout partition string like '(8,4,2):(3,4,1)'.

    Returns:
        tuple: (rank, shape_list, stride_list) or (0, [], []) if partition_str is None
    """

    if partition_str is None:
        return (0, [], [])

    if len(parts := partition_str.split(":")) != 2:
        raise ValueError(f"Invalid partition format: {partition_str}")

    shape_list, stride_list = map(lambda x: list(ast.literal_eval(x)), parts)

    if len(shape_list) != len(stride_list):
        raise ValueError(f"Shape and stride must have same length. Got shape={shape_list}, stride={stride_list}")

    return (len(shape_list), shape_list, stride_list)


def allocate_blocks_vmm(block_sizes, streams, use_vmm=True):
    """
    Allocate memory blocks across devices using stream information.

    This function allocates a series of memory blocks on the devices specified
    by the streams. When VMM is available, it uses CUDA Virtual Memory
    Management for pointer preservation. Otherwise, falls back to separate
    allocations per block using cupy.

    Args:
        block_sizes: List/array of block sizes in bytes
        streams: List of warp.Stream objects, one per block
        use_vmm: If True, try to use VMM; if False, use standard allocation

    Returns:
        List of dicts with 'device_id', 'buffer', 'size', 'offset' for each block

    Example:
        streams = [wp.Stream("cuda:0"), wp.Stream("cuda:1")]
        allocations = allocate_blocks_vmm(
            block_sizes=[256, 512],
            streams=streams
        )

        # Access each allocation
        for alloc in allocations:
            print(f"Device {alloc['device_id']}: buffer size {alloc['size']}")

    Note: VirtualMemoryResource is not yet available in released cuda-python.
          Falls back to cupy allocation.
    """
    if len(block_sizes) != len(streams):
        raise ValueError("block_sizes and streams must have the same length")

    # Try to import VMM if requested
    vmm_available = False
    VirtualMemoryResource = None
    VirtualMemoryResourceOptions = None
    if use_vmm:
        try:
            from cuda.core.experimental import VirtualMemoryResource, VirtualMemoryResourceOptions

            vmm_available = True
        except (ImportError, AttributeError):
            pass

    # Fallback to cupy allocation (VMM not available)
    if not vmm_available:
        import cupy as cp

        allocations = []
        for i, (size, stream) in enumerate(zip(block_sizes, streams)):
            # Get device ordinal directly from stream
            device_ordinal = stream.device.ordinal if hasattr(stream.device, "ordinal") else 0
            device_alias = stream.device.alias if hasattr(stream.device, "alias") else "cuda:0"

            # Use cupy's device context
            with cp.cuda.Device(device_ordinal):
                # Allocate using cupy on the specified device
                arr = cp.empty(size, dtype=cp.uint8)

            allocations.append(
                {
                    "device_id": device_alias,
                    "buffer": arr,  # Keep buffer for fallback path
                    "ptr": arr.data.ptr,  # Also store pointer for consistency
                    "size": size,
                    "offset": 0,
                }
            )
        return allocations

    # VMM path using low-level CUDA driver API
    # Reserve a single virtual address space, then create separate physical
    # allocations for each block, mapping them to their offsets
    import cuda.bindings.driver as cuda_driver
    import cupy as cp

    # Collect all unique devices
    all_device_ordinals = set()
    for stream in streams:
        device_ordinal = stream.device.ordinal if hasattr(stream.device, "ordinal") else 0
        all_device_ordinals.add(device_ordinal)

    num_devices = len(all_device_ordinals)

    # Calculate total size needed for entire allocation
    total_size = sum(block_sizes)

    # Get allocation granularity (use first device as reference)
    first_device_ordinal = list(all_device_ordinals)[0]
    prop = cuda_driver.CUmemAllocationProp()
    prop.type = cuda_driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda_driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = first_device_ordinal

    err, granularity = cuda_driver.cuMemGetAllocationGranularity(
        prop, cuda_driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
    )

    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuMemGetAllocationGranularity failed with error {err}")

    # Round up total size to granularity
    aligned_total_size = ((total_size + granularity - 1) // granularity) * granularity

    print(f"VMM: Reserving {aligned_total_size} bytes of virtual address space")

    # Reserve virtual address space ONCE for the entire allocation
    err, base_dptr = cuda_driver.cuMemAddressReserve(
        aligned_total_size,
        granularity,
        0,  # addr (0 = let CUDA choose)
        0,  # flags
    )

    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuMemAddressReserve failed with error {err}")

    # Convert base pointer to integer for arithmetic
    base_ptr_int = int(base_dptr)
    print(f"VMM: Reserved virtual address space at {hex(base_ptr_int)}")

    # Now create separate physical allocations for each block
    allocations = []
    handles = []  # Keep track of handles for cleanup
    current_offset = 0

    try:
        for i, (size, stream) in enumerate(zip(block_sizes, streams)):
            # Get device for this block
            device_ordinal = stream.device.ordinal if hasattr(stream.device, "ordinal") else 0
            device_alias = stream.device.alias if hasattr(stream.device, "alias") else f"cuda:{device_ordinal}"

            # Round up block size to granularity
            aligned_size = ((size + granularity - 1) // granularity) * granularity

            # Create allocation properties for this device
            block_prop = cuda_driver.CUmemAllocationProp()
            block_prop.type = cuda_driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            block_prop.location.type = cuda_driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            block_prop.location.id = device_ordinal

            # Create physical memory handle for this block
            err, handle = cuda_driver.cuMemCreate(
                aligned_size,
                block_prop,
                0,  # flags
            )

            if err != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuMemCreate failed for block {i} (device {device_ordinal}) with error {err}")

            handles.append((handle, aligned_size))

            print(f"VMM: Created block {i}: {aligned_size} bytes on device {device_ordinal}")

            # Map this physical allocation to the appropriate offset in virtual space
            # Convert to integer, do arithmetic, then use the integer value
            block_vaddr_int = base_ptr_int + current_offset

            err = cuda_driver.cuMemMap(
                block_vaddr_int,
                aligned_size,
                0,  # offset in physical allocation
                handle,
                0,  # flags
            )[0]

            if err != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuMemMap failed for block {i} with error {err}")

            print(f"VMM: Mapped block {i} at offset {current_offset} (vaddr {hex(block_vaddr_int)})")

            # Set access permissions for all devices (enable peer access)
            access_descs = []
            for dev_id in sorted(all_device_ordinals):
                access_desc = cuda_driver.CUmemAccessDesc()
                access_desc.location.type = cuda_driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                access_desc.location.id = dev_id
                access_desc.flags = cuda_driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                access_descs.append(access_desc)

            err = cuda_driver.cuMemSetAccess(block_vaddr_int, aligned_size, access_descs, len(access_descs))[0]

            if err != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuMemSetAccess failed for block {i} with error {err}")

            # Store the allocation info (create cupy array wrapper later when needed)
            allocations.append(
                {"device_id": device_alias, "ptr": block_vaddr_int, "size": size, "offset": current_offset}
            )

            current_offset += aligned_size

        if num_devices > 1:
            print(f"VMM: Enabled peer access across {num_devices} devices")
        print(f"VMM: Successfully allocated {len(allocations)} blocks")

    except Exception as e:
        # Cleanup on failure
        print("VMM: Allocation failed, cleaning up...")
        for handle, size in handles:
            cuda_driver.cuMemRelease(handle)
        cuda_driver.cuMemAddressFree(base_dptr, aligned_total_size)
        raise RuntimeError(f"VMM allocation with low-level API failed: {e}")

    return allocations


def visualize_2d_allocation(global_shape, element_to_place_map, num_places):
    """
    Visualize element allocation for 2D tensors.

    Args:
        global_shape: Tuple of (width, height) or shape with 2 elements
        element_to_place_map: Dictionary mapping element index to place
        num_places: Number of places
    """
    from pycute.core.htuple import leaves

    # Get 2D dimensions
    dims = list(leaves(global_shape))
    if len(dims) != 2:
        print(f"  (Visualization only available for 2D shapes, got {len(dims)}D)")
        return

    width, height = dims

    print(f"\n2D Element Allocation Map ({width}x{height}):")
    print("  Each cell shows which place owns that element")
    print()

    # Create a 2D grid
    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            # Compute linear index (row-major)
            idx = y * width + x
            if idx in element_to_place_map:
                place = element_to_place_map[idx]
                row.append(str(place))
            else:
                row.append(".")
        grid.append(row)

    # Print with borders
    print("  +" + "-" * (width * 2 + 1) + "+")
    for row in grid:
        print("  | " + " ".join(row) + " |")
    print("  +" + "-" * (width * 2 + 1) + "+")
    print()


def allocate_tiled_tensor(tile_shape, tile_dim, partition_desc, streams, dtype, page_size_bytes=2 * 1024 * 1024):
    """
    Allocate a tensor of tiles with localized memory placement.

    This function allocates a distributed tensor where each "element" is actually
    a tile. Memory is allocated using CUDA VMM with page-based locality, where
    pages are assigned to devices based on which place owns the tiles in that page.

    Args:
        tile_shape: Shape of the tile grid (tuple), e.g. (128, 128) for a 128x128 grid of tiles
        tile_dim: Shape of each tile (tuple), e.g. (64, 64)
        partition_desc: PartitionDesc object from wp.blocked() or similar
        streams: List of CUDA streams, one per place
        dtype: Data type (e.g., np.float32, "float32", or numpy dtype)
        page_size_bytes: Size of each memory page (default 2MB)

    Returns:
        cupy.ndarray wrapping the allocated memory, convertible to dlpack

    Example:
        tile_shape = (128, 128)  # 128x128 grid of tiles
        result = wp.blocked(dim=tile_shape, places=8)
        streams = [wp.Stream(f"cuda:{i}") for i in range(len(result.offsets))]
        arr = allocate_tiled_tensor(
            tile_shape=tile_shape,
            tile_dim=(64, 64),
            partition_desc=result,
            streams=streams,
            dtype=np.float32
        )
        wp_arr = wp.from_dlpack(arr.toDlpack())
    """
    import cupy as cp
    import numpy as np

    import warp as wp

    # Map warp dtype to CuPy dtype for scalar types (similar to empty_managed)
    dtype_map = {
        float: cp.float32,
        wp.float32: cp.float32,
        wp.float64: cp.float64,
        int: cp.int32,
        wp.int32: cp.int32,
        wp.int64: cp.int64,
        wp.uint32: cp.uint32,
        wp.uint64: cp.uint64,
    }

    # Convert dtype to CuPy dtype
    is_structured_type = False
    if dtype in dtype_map:
        cp_dtype = dtype_map[dtype]
        np_dtype = cp.dtype(cp_dtype)
        elem_size_bytes = np_dtype.itemsize
    elif isinstance(dtype, str):
        cp_dtype = dtype  # String dtype
        np_dtype = np.dtype(dtype)
        elem_size_bytes = np_dtype.itemsize
    elif hasattr(dtype, "dtype"):
        # Handle numpy/cupy dtype objects (e.g., np.float32, cp.float32)
        np_dtype = np.dtype(dtype)
        cp_dtype = np_dtype
        elem_size_bytes = np_dtype.itemsize
    elif isinstance(dtype, np.dtype):
        # Already a numpy dtype
        np_dtype = dtype
        cp_dtype = dtype
        elem_size_bytes = np_dtype.itemsize
    elif dtype in [np.float32, np.float64, np.int32, np.int64, np.uint32, np.uint64]:
        # Handle numpy dtype classes directly
        np_dtype = np.dtype(dtype)
        cp_dtype = np_dtype
        elem_size_bytes = np_dtype.itemsize
    else:
        # Structured type (vec2, vec3, mat22, etc.)
        # CuPy can't handle Warp's structured types, so allocate as bytes
        is_structured_type = True
        try:
            elem_size_bytes = wp.types.type_size_in_bytes(dtype)
        except:
            raise ValueError(f"Cannot determine size for dtype {dtype}")
        # Allocate as uint8 (bytes) and let the wrapper create the proper warp array
        cp_dtype = cp.uint8
        np_dtype = cp.dtype(cp.uint8)

    # Validate partition_desc has the necessary fields
    if partition_desc.partition_cute is None or partition_desc.offset_layout is None:
        raise ValueError("partition_desc must have partition_cute and offset_layout computed")

    # Compute tile size
    tile_size = int(np.prod(tile_dim))
    tile_size_bytes = tile_size * elem_size_bytes

    # Compute total tiles and elements from explicit tile_shape
    total_tiles = int(np.prod(tile_shape))
    total_elements = total_tiles * tile_size

    # Compute global shape in elements by multiplying tile_shape by tile_dim
    # e.g., tile_shape=(128, 128), tile_dim=(64, 64) -> global_shape=(8192, 8192)
    if isinstance(tile_shape, (list, tuple)):
        global_shape = tuple(ts * td for ts, td in zip(tile_shape, tile_dim))
    else:
        # 1D case
        global_shape = (tile_shape * tile_dim[0],)

    # Store the logical shape (for structured types, this is different from CuPy array shape)
    logical_shape = global_shape

    print(f"\n{'=' * 60}")
    print("TILED TENSOR ALLOCATION")
    print(f"{'=' * 60}")
    print(f"Tile shape:       {tile_shape}")
    print(f"Tile dimensions:  {tile_dim}")
    print(f"Tile size:        {tile_size} elements = {tile_size_bytes} bytes")
    print(f"Element dtype:    {np_dtype}")
    print(f"Element size:     {elem_size_bytes} bytes")
    print(f"Total tiles:      {total_tiles}")
    print(f"Total elements:   {total_elements}")
    print(f"Global shape:     {global_shape}")
    print(f"Number of places: {len(partition_desc.offsets)}")
    print(f"Page size:        {page_size_bytes} bytes")
    print()

    # Compute total footprint
    footprint_bytes = total_elements * elem_size_bytes
    num_pages = (footprint_bytes + page_size_bytes - 1) // page_size_bytes

    print(f"Total footprint:  {footprint_bytes} bytes ({footprint_bytes / (1024**2):.2f} MB)")
    print(f"Number of pages:  {num_pages}")
    print()

    # For structured types, adjust the CuPy array shape to be in bytes
    if is_structured_type:
        # CuPy array will be 1D bytes, warp array will have the logical_shape
        global_shape = (footprint_bytes,)

    # Determine page ownership using partition_desc.get_element_owner
    print("Computing page ownership...")
    page_owners = []
    samples_per_page = 5

    for page_idx in range(num_pages):
        page_start_byte = page_idx * page_size_bytes
        page_end_byte = min(page_start_byte + page_size_bytes, footprint_bytes)
        page_size = page_end_byte - page_start_byte

        # Sample tiles in this page to determine ownership
        owner_votes = {}
        for _ in range(samples_per_page):
            # Random byte offset within page
            sample_byte = random.randint(0, page_size - 1)
            global_byte = page_start_byte + sample_byte

            # Convert to element index
            elem_idx = global_byte // elem_size_bytes

            # Convert element index to tile index (always 1D reasoning)
            # The partition works with flattened tile indices
            tile_idx = elem_idx // tile_size

            # Get owner using partition_desc (partition takes 1D tile index)
            owner = partition_desc.get_element_owner(tile_idx)
            if owner is None:
                raise ValueError(f"No owner found for tile {tile_idx} (element {elem_idx}, byte {global_byte})")
            owner_votes[owner] = owner_votes.get(owner, 0) + 1

        # Assign page to majority owner
        if owner_votes:
            page_owner = max(owner_votes.items(), key=lambda x: x[1])[0]
        else:
            page_owner = 0  # Default to first place
            raise ValueError(f"No owner found for page {page_idx}")

        page_owners.append(page_owner)

    print("✓ Page ownership computed")
    print()

    # Merge contiguous pages assigned to the same place
    contiguous_parts = []
    if num_pages > 0:
        current_place = page_owners[0]
        start_page = 0

        for i in range(1, num_pages):
            if page_owners[i] != current_place:
                # End of contiguous region
                contiguous_parts.append(
                    {
                        "place": current_place,
                        "start_page": start_page,
                        "end_page": i,
                        "num_pages": i - start_page,
                        "start_byte": start_page * page_size_bytes,
                        "end_byte": min(i * page_size_bytes, footprint_bytes),
                    }
                )
                # Start new region
                current_place = page_owners[i]
                start_page = i

        # Add the final region
        contiguous_parts.append(
            {
                "place": current_place,
                "start_page": start_page,
                "end_page": num_pages,
                "num_pages": num_pages - start_page,
                "start_byte": start_page * page_size_bytes,
                "end_byte": footprint_bytes,
            }
        )

    print(f"Contiguous memory regions: {len(contiguous_parts)}")
    for i, part in enumerate(contiguous_parts[:10]):  # Show first 10
        print(
            f"  Part {i}: Place {part['place']}, Pages {part['num_pages']}, "
            f"Bytes [{part['start_byte']} - {part['end_byte']})"
        )
    if len(contiguous_parts) > 10:
        print(f"  ... ({len(contiguous_parts) - 10} more parts)")
    print()

    # Allocate memory using VMM
    print("Allocating memory with CUDA VMM...")

    block_sizes = []
    block_streams = []

    for part in contiguous_parts:
        place_id = part["place"]
        if place_id >= 0 and place_id < len(streams):
            block_size = part["end_byte"] - part["start_byte"]

            block_sizes.append(block_size)
            block_streams.append(streams[place_id])

    # Allocate the blocks
    try:
        vmm_allocations = allocate_blocks_vmm(block_sizes, block_streams)
        print(f"✓ Allocated {len(vmm_allocations)} memory blocks")
    except Exception as e:
        import sys
        import traceback

        print(f"✗ VMM allocation failed: {e}")
        if False:  # Set to True for detailed debugging
            print("Full traceback:")
            traceback.print_exception(type(e), e, e.__traceback__, file=sys.stdout)
        print("Falling back to simple allocation...")
        # Fallback: allocate on first device
        from cupy.cuda import memory

        memptr = memory.malloc_managed(footprint_bytes)
        if is_structured_type:
            # For structured types, create 1D byte array
            cupy_arr = cp.ndarray((footprint_bytes,), dtype=cp.uint8, memptr=memptr)
        else:
            cupy_arr = cp.ndarray(global_shape, dtype=cp_dtype, memptr=memptr)
        print(f"✓ Allocated {footprint_bytes} bytes on managed memory")
        return cupy_arr

    # Create a unified memory view
    # When VMM is used, all allocations are mapped into a single contiguous virtual
    # address space starting at the base pointer (first allocation's ptr).
    # Even if there are multiple physical allocations (vmm_allocations list has multiple entries),
    # they form ONE contiguous virtual buffer.

    # Check if this is a VMM allocation by looking for the "ptr" field
    # (VMM allocations have "ptr", fallback allocations have "buffer")
    is_vmm = len(vmm_allocations) > 0 and "ptr" in vmm_allocations[0]

    if is_vmm:
        # VMM: All blocks are mapped into a single contiguous virtual address space
        # The base pointer is the first allocation's pointer
        base_ptr = vmm_allocations[0]["ptr"]

        # Calculate total size across all VMM allocations
        # (should equal footprint_bytes, but let's be explicit)
        total_vmm_size = sum(alloc["size"] for alloc in vmm_allocations)

        # Sanity check
        if total_vmm_size != footprint_bytes:
            print(f"⚠ Warning: VMM total size ({total_vmm_size}) != footprint ({footprint_bytes})")

        # Create cupy array wrapper around the entire VMM virtual address space
        mem = cp.cuda.UnownedMemory(base_ptr, footprint_bytes, owner=None)
        memptr = cp.cuda.MemoryPointer(mem, 0)

        if is_structured_type:
            # For structured types, create 1D byte array
            cupy_arr = cp.ndarray((footprint_bytes,), dtype=cp.uint8, memptr=memptr)
        else:
            # For scalar types, create array with target dtype and shape
            cupy_arr = cp.ndarray(global_shape, dtype=cp_dtype, memptr=memptr)

        print(
            f"✓ Using VMM allocated memory: {len(vmm_allocations)} physical blocks mapped to single virtual address space"
        )
        print(f"  Base pointer: {hex(base_ptr)}")
        print(f"  Total size: {footprint_bytes} bytes")
        print(f"  Array shape: {cupy_arr.shape}")
        print(f"{'=' * 60}")
        print()
        return cupy_arr

    # Non-VMM fallback: separate allocations
    # Fall back to managed memory for simplicity
    print("Non-VMM allocations detected, creating unified managed memory view...")
    from cupy.cuda import memory

    memptr = memory.malloc_managed(footprint_bytes)
    if is_structured_type:
        # For structured types, create 1D byte array
        cupy_arr = cp.ndarray((footprint_bytes,), dtype=cp.uint8, memptr=memptr)
    else:
        cupy_arr = cp.ndarray(global_shape, dtype=cp_dtype, memptr=memptr)

    print(f"✓ Created cupy array with shape {cupy_arr.shape}")
    print(f"{'=' * 60}")
    print()

    return cupy_arr


def empty(shape, partition_desc, streams, dtype=float, tile_dim=None, page_size_bytes=2 * 1024 * 1024):
    """
    Allocate an uninitialized array with localized memory placement across devices.

    This function allocates memory distributed across multiple devices using CUDA VMM,
    with page-based locality for optimal multi-GPU performance.

    Args:
        shape: Global shape of the array in elements (e.g., (8192, 8192))
        partition_desc: Either a PartitionDesc object or a policy function (dim, streams) -> PartitionDesc
        streams: List of CUDA streams, one per place
        dtype: Data type (default: float)
        tile_dim: Shape of each tile (default: None, which means element-level distribution with all 1s)
        page_size_bytes: Size of each memory page (default 2MB)

    Returns:
        warp.array with the allocated memory (uninitialized)

    Examples:
        # Element-level distribution (default)
        policy = wp.blocked()
        streams = [wp.Stream(f"cuda:{i}") for i in range(8)]
        arr = wp.localized.empty(
            shape=(8192, 8192),
            partition_desc=policy,
            streams=streams,
            dtype=float
        )

        # With explicit tile size for coarser-grained distribution
        arr = wp.localized.empty(
            shape=(8192, 8192),
            tile_dim=(64, 64),
            partition_desc=policy,
            streams=streams,
            dtype=wp.vec2
        )
    """
    # Default tile_dim to all 1s if not specified (element-level distribution)
    if tile_dim is None:
        if isinstance(shape, (list, tuple)):
            tile_dim = tuple(1 for _ in shape)
        else:
            tile_dim = (1,)

    # Compute tile shape from global shape and tile dimensions
    if isinstance(shape, (list, tuple)) and isinstance(tile_dim, (list, tuple)):
        tile_shape = tuple(s // td for s, td in zip(shape, tile_dim))
    else:
        tile_shape = shape // tile_dim

    # If partition_desc is a callable (policy function), call it to get PartitionDesc
    if callable(partition_desc):
        partition_desc = partition_desc(dim=tile_shape, streams=streams)

    # Allocate the tiled tensor
    cupy_arr = allocate_tiled_tensor(
        tile_shape=tile_shape,
        tile_dim=tile_dim,
        partition_desc=partition_desc,
        streams=streams,
        dtype=dtype,
        page_size_bytes=page_size_bytes,
    )

    # Convert to warp array
    import numpy as np

    import warp as wp

    # Check if dtype is a structured type (vec2, vec3, mat22, etc.)
    # Structured types can't be exported via DLPack
    dtype_map = {
        float: wp.float32,
        wp.float32: wp.float32,
        wp.float64: wp.float64,
        int: wp.int32,
        wp.int32: wp.int32,
        wp.int64: wp.int64,
        wp.uint32: wp.uint32,
        wp.uint64: wp.uint64,
    }

    is_scalar_type = dtype in dtype_map

    # Also check if it's a numpy/cupy dtype
    if not is_scalar_type:
        if hasattr(dtype, "dtype") or isinstance(dtype, np.dtype):
            is_scalar_type = True
        elif dtype in [np.float32, np.float64, np.int32, np.int64, np.uint32, np.uint64]:
            is_scalar_type = True

    if is_scalar_type:
        # Simple scalar type - use DLPack conversion
        return wp.from_dlpack(cupy_arr)
    else:
        # Structured type (vec2, vec3, mat22, etc.) - create warp array directly from pointer
        # Get the device from the first stream
        device = streams[0].device if streams else "cuda:0"

        # Create warp array directly with the CuPy array's memory pointer
        arr = wp.array(ptr=cupy_arr.data.ptr, shape=shape, dtype=dtype, device=device, copy=False)

        # Attach the CuPy array to the warp array to keep it alive
        arr._cupy_arr = cupy_arr

        return arr


def zeros(shape, partition_desc, streams, dtype=float, tile_dim=None, page_size_bytes=2 * 1024 * 1024):
    """
    Allocate a zero-initialized array with localized memory placement across devices.

    This function allocates memory distributed across multiple devices using CUDA VMM,
    with page-based locality for optimal multi-GPU performance.

    Args:
        shape: Global shape of the array in elements (e.g., (8192, 8192))
        partition_desc: Either a PartitionDesc object or a policy function (dim, streams) -> PartitionDesc
        streams: List of CUDA streams, one per place
        dtype: Data type (default: float)
        tile_dim: Shape of each tile (default: None, which means element-level distribution with all 1s)
        page_size_bytes: Size of each memory page (default 2MB)

    Returns:
        warp.array with the allocated memory (initialized to zero)

    Examples:
        # Element-level distribution (default)
        policy = wp.blocked()
        streams = [wp.Stream(f"cuda:{i}") for i in range(8)]
        arr = wp.localized.zeros(
            shape=(8192, 8192),
            partition_desc=policy,
            streams=streams,
            dtype=float
        )

        # With explicit tile size for coarser-grained distribution
        arr = wp.localized.zeros(
            shape=(8192, 8192),
            tile_dim=(64, 64),
            partition_desc=policy,
            streams=streams,
            dtype=wp.vec2
        )
    """
    # Default tile_dim to all 1s if not specified (element-level distribution)
    if tile_dim is None:
        if isinstance(shape, (list, tuple)):
            tile_dim = tuple(1 for _ in shape)
        else:
            tile_dim = (1,)

    # Compute tile shape from global shape and tile dimensions
    if isinstance(shape, (list, tuple)) and isinstance(tile_dim, (list, tuple)):
        tile_shape = tuple(s // td for s, td in zip(shape, tile_dim))
    else:
        tile_shape = shape // tile_dim

    # If partition_desc is a callable (policy function), call it to get PartitionDesc
    if callable(partition_desc):
        partition_desc = partition_desc(dim=tile_shape, streams=streams)

    # Allocate the tiled tensor
    cupy_arr = allocate_tiled_tensor(
        tile_shape=tile_shape,
        tile_dim=tile_dim,
        partition_desc=partition_desc,
        streams=streams,
        dtype=dtype,
        page_size_bytes=page_size_bytes,
    )

    # Initialize to zero
    cupy_arr.fill(0)

    # Convert to warp array
    import warp as wp

    # Check if dtype is a structured type (vec2, vec3, mat22, etc.)
    # Structured types can't be exported via DLPack
    dtype_map = {
        float: wp.float32,
        wp.float32: wp.float32,
        wp.float64: wp.float64,
        int: wp.int32,
        wp.int32: wp.int32,
        wp.int64: wp.int64,
        wp.uint32: wp.uint32,
        wp.uint64: wp.uint64,
    }

    is_scalar_type = dtype in dtype_map

    if is_scalar_type:
        # Simple scalar type - use DLPack conversion
        return wp.from_dlpack(cupy_arr)
    else:
        # Structured type (vec2, vec3, mat22, etc.) - create warp array directly from pointer
        # Get the device from the first stream
        device = streams[0].device if streams else "cuda:0"

        # Create warp array directly with the CuPy array's memory pointer
        arr = wp.array(ptr=cupy_arr.data.ptr, shape=shape, dtype=dtype, device=device, copy=False)

        # Attach the CuPy array to the warp array to keep it alive
        arr._cupy_arr = cupy_arr

        return arr


def empty_managed(
    shape: int | tuple[int, ...] | list[int],
    dtype: type = float,
    device: str = "cuda:0",
):
    """
    Allocate an uninitialized array using CUDA managed memory (unified memory).

    Managed memory provides automatic memory migration between host and device,
    allowing the same memory address to be accessed from both CPU and GPU.
    This simplifies memory management in multi-GPU scenarios but may have
    performance implications compared to explicit memory placement.

    Args:
        shape: Array dimensions (e.g., (8192, 8192) or 8192)
        dtype: Data type (default: float). Must match the CuPy dtype exactly.
        device: Device to associate with the array (default: "cuda:0")

    Returns:
        warp.array backed by managed memory (uninitialized)

    Examples:
        # Create a managed 2D array
        arr = wp.empty_managed(shape=(1024, 1024), dtype=float)

        # Create a managed 1D array
        arr = wp.empty_managed(shape=1024, dtype=float)

        # Use with specific dtype
        arr = wp.empty_managed(shape=(512, 512), dtype=wp.float32)

    Note:
        This function requires CuPy to be installed and uses CuPy's managed memory
        allocator under the hood. The array is accessible from all CUDA devices
        and the CPU without explicit memory transfers.
    """
    import numpy as np

    import warp as wp

    try:
        import cupy as cp
        from cupy.cuda import memory
    except ImportError:
        raise ImportError("CuPy is required to use managed memory. Install it with: pip install cupy-cuda12x")

    # Normalize shape to tuple
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)

    # Map warp dtype to CuPy dtype for scalar types
    dtype_map = {
        float: cp.float32,
        wp.float32: cp.float32,
        wp.float64: cp.float64,
        int: cp.int32,
        wp.int32: cp.int32,
        wp.int64: cp.int64,
        wp.uint32: cp.uint32,
        wp.uint64: cp.uint64,
    }

    # Determine if this is a scalar or structured type
    is_scalar_type = dtype in dtype_map
    cupy_dtype = None

    if is_scalar_type:
        # Simple scalar type - use DLPack conversion
        cupy_dtype = dtype_map[dtype]
        element_size = cp.dtype(cupy_dtype).itemsize
    elif hasattr(dtype, "dtype") or isinstance(dtype, np.dtype):
        # Handle numpy/cupy dtype objects (e.g., np.float32, cp.float32)
        cupy_dtype = np.dtype(dtype)
        element_size = cupy_dtype.itemsize
        is_scalar_type = True
    elif dtype in [np.float32, np.float64, np.int32, np.int64, np.uint32, np.uint64]:
        # Handle numpy dtype classes directly
        cupy_dtype = np.dtype(dtype)
        element_size = cupy_dtype.itemsize
        is_scalar_type = True

    if is_scalar_type and cupy_dtype is not None:
        # Simple scalar type - use DLPack conversion

        nbytes = 1
        for dim in shape:
            nbytes *= dim
        nbytes *= element_size

        # Allocate managed memory
        memptr = memory.malloc_managed(nbytes)

        # Create CuPy array with managed memory
        cupy_arr = cp.ndarray(shape, dtype=cupy_dtype, memptr=memptr)

        # Convert to warp array via DLPack
        return wp.from_dlpack(cupy_arr.toDlpack())
    else:
        # Structured type (vec2, vec3, mat22, etc.) - use direct ptr allocation
        try:
            element_size = wp.types.type_size_in_bytes(dtype)
        except:
            raise ValueError(f"Cannot determine size for dtype {dtype}")

        nbytes = 1
        for dim in shape:
            nbytes *= dim
        nbytes *= element_size

        # Allocate managed memory
        memptr = memory.malloc_managed(nbytes)

        # Create warp array directly with the managed memory pointer
        # Store a reference to memptr to prevent garbage collection
        arr = wp.array(ptr=memptr.ptr, shape=shape, dtype=dtype, device=device, copy=False)

        # Attach the CuPy memory object to the warp array to keep it alive
        arr._cupy_memptr = memptr

        return arr


def zeros_managed(
    shape: int | tuple[int, ...] | list[int],
    dtype: type = float,
    device: str = "cuda:0",
):
    """
    Allocate a zero-initialized array using CUDA managed memory (unified memory).

    Managed memory provides automatic memory migration between host and device,
    allowing the same memory address to be accessed from both CPU and GPU.
    This simplifies memory management in multi-GPU scenarios but may have
    performance implications compared to explicit memory placement.

    Args:
        shape: Array dimensions (e.g., (8192, 8192) or 8192)
        dtype: Data type (default: float). Must match the CuPy dtype exactly.
        device: Device to associate with the array (default: "cuda:0")

    Returns:
        warp.array backed by managed memory (initialized to zero)

    Examples:
        # Create a zero-initialized managed 2D array
        arr = wp.zeros_managed(shape=(1024, 1024), dtype=float)

        # Create a zero-initialized managed 1D array
        arr = wp.zeros_managed(shape=1024, dtype=float)

        # Use with specific dtype
        arr = wp.zeros_managed(shape=(512, 512), dtype=wp.float32)

        # Use in multi-GPU context
        managedA = wp.zeros_managed((nx, ny), dtype=float)
        wp.launch(kernel, dim=(nx, ny), outputs=[managedA], device="cuda:0")

    Note:
        This function requires CuPy to be installed and uses CuPy's managed memory
        allocator under the hood. The array is accessible from all CUDA devices
        and the CPU without explicit memory transfers.
    """
    arr = empty_managed(shape=shape, dtype=dtype, device=device)

    # Initialize to zero
    # Get the underlying CuPy array from DLPack
    # We need to zero the memory - we can do this via the device
    arr.zero_()

    return arr


def launch(kernel, dim, inputs=None, outputs=None, primary_stream=None, mapping=None, streams=None, **kwargs):
    """
    Launch a kernel across multiple devices with localized data placement.

    This function launches a kernel with work (blocks) distributed according to a partition
    policy (mapping). Each partition is launched on its corresponding stream/device.

    Args:
        kernel: The kernel function to launch
        dim: Tuple or int specifying the launch dimensions (e.g., (1024, 1024) for 1024x1024 threads)
        inputs: List of input arrays (optional)
        outputs: List of output arrays (optional)
        primary_stream: Primary stream for synchronization (default: streams[0])
        mapping: Either a PartitionDesc object or a policy function (dim, streams) -> PartitionDesc
        streams: List of streams, one per place in the mapping
        **kwargs: Additional arguments passed to wp.launch (e.g., block_dim, max_blocks)

    Examples:
        # Direct mode with PartitionDesc
        result = wp.blocked(dim=(1024, 1024), places=8)
        streams = [wp.Stream(f"cuda:{i % ndevices}") for i in range(len(result.offsets))]
        wp.localized.launch(
            kernel=my_kernel,
            dim=(1024, 1024),
            inputs=[A],
            outputs=[C],
            mapping=result,
            streams=streams
        )

        # Policy mode
        policy = wp.blocked()
        streams = [wp.Stream(f"cuda:{i % ndevices}") for i in range(8)]
        wp.localized.launch(
            kernel=my_kernel,
            dim=(1024, 1024),
            inputs=[A],
            outputs=[C],
            mapping=policy,
            streams=streams
        )
    """
    import warp as wp

    if mapping is None:
        raise ValueError("mapping (PartitionDesc or policy function) must be provided")
    if streams is None:
        raise ValueError("streams must be provided")

    # Get block_dim from kwargs (default 256)
    block_dim_kwarg = kwargs.get("block_dim", 256)

    # Convert thread dimensions to block dimensions for partitioning
    # For launch_localized, we partition CUDA blocks, not threads
    # CUDA grids are always 1D (even for multi-dimensional thread launches)
    if isinstance(dim, int):
        total_threads = dim
    else:
        # Compute total threads across all dimensions
        total_threads = 1
        for d in dim:
            total_threads *= d

    # Compute the number of 1D blocks needed
    total_blocks = (total_threads + block_dim_kwarg - 1) // block_dim_kwarg

    # If mapping is a callable (policy function), call it with 1D block count
    # The partition distributes the 1D block grid across streams/devices
    if callable(mapping):
        mapping = mapping(dim=(total_blocks,), streams=streams)

    # Default primary stream to first stream
    if primary_stream is None:
        primary_stream = streams[0]

    # Ensure inputs and outputs are lists, not None
    if inputs is None:
        inputs = []
    if outputs is None:
        outputs = []

    # Validate that we have the right number of streams
    if len(streams) != len(mapping.offsets):
        raise ValueError(f"Number of streams ({len(streams)}) must match number of places ({len(mapping.offsets)})")

    # Step 1: Record event on primary stream
    e0 = wp.Event(device=primary_stream.device)
    primary_stream.record_event(e0)

    # Step 2: All other streams wait for primary
    for stream in streams:
        if stream != primary_stream:
            stream.wait_event(e0)

    # Step 3: Launch kernels on all places in parallel
    for place_idx, offset in enumerate(mapping.offsets):
        stream = streams[place_idx]

        wp.launch(
            kernel,
            dim=dim,
            inputs=inputs,
            outputs=outputs,
            partition=mapping.partition,
            offset=offset,
            stream=stream,
            **kwargs,
        )

    # Step 4-5: Other streams signal completion, primary waits
    for stream in streams:
        if stream != primary_stream:
            ei = wp.Event(device=stream.device)
            stream.record_event(ei)
            primary_stream.wait_event(ei)


def launch_tiled(
    kernel, dim, inputs=None, outputs=None, primary_stream=None, block_dim=None, mapping=None, streams=None, **kwargs
):
    """
    Launch a tiled kernel across multiple devices with localized data placement.

    This function launches a tiled kernel with work distributed according to a partition
    policy (mapping). Each partition is launched on its corresponding stream/device.

    Args:
        kernel: The kernel function to launch
        dim: Tuple specifying the tile grid dimensions (e.g., (128, 128) for 128x128 tiles)
        inputs: List of input arrays (optional)
        outputs: List of output arrays (optional)
        primary_stream: Primary stream for synchronization (default: streams[0])
        block_dim: Thread block dimension for the kernel
        mapping: Either a PartitionDesc object or a policy function (dim, streams) -> PartitionDesc
        streams: List of streams, one per place in the mapping
        **kwargs: Additional arguments passed to wp.launch_tiled

    Examples:
        # Direct mode with PartitionDesc
        result = wp.blocked(dim=(128, 128), places=8)
        streams = [wp.Stream(f"cuda:{i % ndevices}") for i in range(len(result.offsets))]
        wp.localized.launch_tiled(
            kernel=my_kernel,
            dim=(128, 128),
            inputs=[A],
            outputs=[C],
            block_dim=32,
            mapping=result,
            streams=streams
        )

        # Policy mode (new)
        policy = wp.blocked()
        streams = [wp.Stream(f"cuda:{i % ndevices}") for i in range(8)]
        wp.localized.launch_tiled(
            kernel=my_kernel,
            dim=(128, 128),
            inputs=[A],
            outputs=[C],
            block_dim=32,
            mapping=policy,
            streams=streams
        )
    """
    import warp as wp

    if mapping is None:
        raise ValueError("mapping (PartitionDesc or policy function) must be provided")
    if streams is None:
        raise ValueError("streams must be provided")

    # If mapping is a callable (policy function), call it to get PartitionDesc
    if callable(mapping):
        mapping = mapping(dim=dim, streams=streams)

    # Default primary stream to first stream
    if primary_stream is None:
        primary_stream = streams[0]

    # Ensure inputs and outputs are lists, not None
    if inputs is None:
        inputs = []
    if outputs is None:
        outputs = []

    # Validate that we have the right number of streams
    if len(streams) != len(mapping.offsets):
        raise ValueError(f"Number of streams ({len(streams)}) must match number of places ({len(mapping.offsets)})")

    # Step 1: Record event on primary stream
    e0 = wp.Event(device=primary_stream.device)
    primary_stream.record_event(e0)

    # Step 2: All other streams wait for primary
    for stream in streams:
        if stream != primary_stream:
            stream.wait_event(e0)

    # Step 3: Launch kernels on all places in parallel
    for place_idx, offset in enumerate(mapping.offsets):
        stream = streams[place_idx]

        wp.launch_tiled(
            kernel,
            dim=dim,
            inputs=inputs,
            outputs=outputs,
            partition=mapping.partition,
            offset=offset,
            stream=stream,
            block_dim=block_dim,
            **kwargs,
        )

    # Step 4-5: Other streams signal completion, primary waits
    for stream in streams:
        if stream != primary_stream:
            ei = wp.Event(device=stream.device)
            stream.record_event(ei)
            primary_stream.wait_event(ei)
