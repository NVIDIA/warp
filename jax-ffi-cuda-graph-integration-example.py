import ctypes
import enum
import math
import warp as wp
import jax
import jax.numpy as jp

#######################################################################
# ctypes structures and enums for XLA's FFI API:
# https://github.com/openxla/xla/blob/a1a5e62fbffa3a3b6c409d72607456cf5b353a22/xla/ffi/api/c_api.h
#######################################################################

# typedef enum {
#   XLA_FFI_Extension_Metadata = 1,
# } XLA_FFI_Extension_Type;
class XLA_FFI_Extension_Type(enum.IntEnum):
    Metadata = 1

# typedef struct XLA_FFI_Extension_Base {
#   size_t struct_size;
#   XLA_FFI_Extension_Type type;
#   struct XLA_FFI_Extension_Base* next;
# } XLA_FFI_Extension_Base;
class XLA_FFI_Extension_Base(ctypes.Structure):
    pass

XLA_FFI_Extension_Base._fields_ = [
    ("struct_size", ctypes.c_size_t),
    ("type", ctypes.c_int), # XLA_FFI_Extension_Type
    ("next", ctypes.POINTER(XLA_FFI_Extension_Base))
]

# typedef enum {
#   XLA_FFI_ExecutionStage_INSTANTIATE = 0,
#   XLA_FFI_ExecutionStage_PREPARE = 1,
#   XLA_FFI_ExecutionStage_INITIALIZE = 2,
#   XLA_FFI_ExecutionStage_EXECUTE = 3,
# } XLA_FFI_ExecutionStage;
class XLA_FFI_ExecutionStage(enum.IntEnum):
    INSTANTIATE = 0
    PREPARE = 1
    INITIALIZE = 2
    EXECUTE = 3

# typedef enum {
#   XLA_FFI_DataType_INVALID = 0,
#   XLA_FFI_DataType_PRED = 1,
#   XLA_FFI_DataType_S8 = 2,
#   XLA_FFI_DataType_S16 = 3,
#   XLA_FFI_DataType_S32 = 4,
#   XLA_FFI_DataType_S64 = 5,
#   XLA_FFI_DataType_U8 = 6,
#   XLA_FFI_DataType_U16 = 7,
#   XLA_FFI_DataType_U32 = 8,
#   XLA_FFI_DataType_U64 = 9,
#   XLA_FFI_DataType_F16 = 10,
#   XLA_FFI_DataType_F32 = 11,
#   XLA_FFI_DataType_F64 = 12,
#   XLA_FFI_DataType_BF16 = 16,
#   XLA_FFI_DataType_C64 = 15,
#   XLA_FFI_DataType_C128 = 18,
#   XLA_FFI_DataType_TOKEN = 17,
#   XLA_FFI_DataType_F8E5M2 = 19,
#   XLA_FFI_DataType_F8E3M4 = 29,
#   XLA_FFI_DataType_F8E4M3 = 28,
#   XLA_FFI_DataType_F8E4M3FN = 20,
#   XLA_FFI_DataType_F8E4M3B11FNUZ = 23,
#   XLA_FFI_DataType_F8E5M2FNUZ = 24,
#   XLA_FFI_DataType_F8E4M3FNUZ = 25,
#   XLA_FFI_DataType_F4E2M1FN = 32,
#   XLA_FFI_DataType_F8E8M0FNU = 33,
# } XLA_FFI_DataType;
class XLA_FFI_DataType(enum.IntEnum):
    INVALID = 0
    PRED = 1
    S8 = 2
    S16 = 3
    S32 = 4
    S64 = 5
    U8 = 6
    U16 = 7
    U32 = 8
    U64 = 9
    F16 = 10
    F32 = 11
    F64 = 12
    BF16 = 16
    C64 = 15
    C128 = 18
    TOKEN = 17
    F8E5M2 = 19
    F8E3M4 = 29
    F8E4M3 = 28
    F8E4M3FN = 20
    F8E4M3B11FNUZ = 23
    F8E5M2FNUZ = 24
    F8E4M3FNUZ = 25
    F4E2M1FN = 32
    F8E8M0FNU = 33

# struct XLA_FFI_Buffer {
#   size_t struct_size;
#   XLA_FFI_Extension_Base* extension_start;
#
#   XLA_FFI_DataType dtype;
#   void* data;
#   int64_t rank;
#   int64_t* dims;  // length == rank
# };
class XLA_FFI_Buffer(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_size_t),
                ("extension_start", ctypes.POINTER(XLA_FFI_Extension_Base)),
                ("dtype", ctypes.c_int), # XLA_FFI_DataType
                ("data", ctypes.c_void_p),
                ("rank", ctypes.c_int64),
                ("dims", ctypes.POINTER(ctypes.c_int64))
    ]

# typedef enum {
#   XLA_FFI_ArgType_BUFFER = 1,
# } XLA_FFI_ArgType;
class XLA_FFI_ArgType(enum.IntEnum):
    BUFFER = 1

# typedef enum {
#   XLA_FFI_RetType_BUFFER = 1,
# } XLA_FFI_RetType;
class XLA_FFI_RetType(enum.IntEnum):
    BUFFER = 1

# struct XLA_FFI_Args {
#   size_t struct_size;
#   XLA_FFI_Extension_Base* extension_start;
#   int64_t size;
#   XLA_FFI_ArgType* types;  // length == size
#   void** args;             // length == size
# };
class XLA_FFI_Args(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_size_t),
                ("extension_start", ctypes.POINTER(XLA_FFI_Extension_Base)),
                ("size", ctypes.c_int64),
                ("types", ctypes.POINTER(ctypes.c_int)), # XLA_FFI_ArgType*
                ("args", ctypes.POINTER(ctypes.c_void_p))
    ]

# struct XLA_FFI_Rets {
#   size_t struct_size;
#   XLA_FFI_Extension_Base* extension_start;
#   int64_t size;
#   XLA_FFI_RetType* types;  // length == size
#   void** rets;             // length == size
# };
class XLA_FFI_Rets(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_size_t),
                ("extension_start", ctypes.POINTER(XLA_FFI_Extension_Base)),
                ("size", ctypes.c_int64),
                ("types", ctypes.POINTER(ctypes.c_int)), # XLA_FFI_RetType*
                ("rets", ctypes.POINTER(ctypes.c_void_p))
    ]

# typedef struct XLA_FFI_ByteSpan {
#   const char* ptr;
#   size_t len;
# } XLA_FFI_ByteSpan;
class XLA_FFI_ByteSpan(ctypes.Structure):
    _fields_ = [("ptr", ctypes.POINTER(ctypes.c_char)),
                ("len", ctypes.c_size_t)]

# typedef enum {
#   XLA_FFI_AttrType_ARRAY = 1,
#   XLA_FFI_AttrType_DICTIONARY = 2,
#   XLA_FFI_AttrType_SCALAR = 3,
#   XLA_FFI_AttrType_STRING = 4,
# } XLA_FFI_AttrType;
class XLA_FFI_AttrType(enum.IntEnum):
    ARRAY = 1
    DICTIONARY = 2
    SCALAR = 3
    STRING = 4

# struct XLA_FFI_Attrs {
#   size_t struct_size;
#   XLA_FFI_Extension_Base* extension_start;
#   int64_t size;
#   XLA_FFI_AttrType* types;   // length == size
#   XLA_FFI_ByteSpan** names;  // length == size
#   void** attrs;              // length == size
# };
class XLA_FFI_Attrs(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_size_t),
                ("extension_start", ctypes.POINTER(XLA_FFI_Extension_Base)),
                ("size", ctypes.c_int64),
                ("types", ctypes.POINTER(ctypes.c_int)), # XLA_FFI_AttrType*
                ("names", ctypes.POINTER(ctypes.POINTER(XLA_FFI_ByteSpan))),
                ("attrs", ctypes.POINTER(ctypes.c_void_p))
    ]

# struct XLA_FFI_Api_Version {
#   size_t struct_size;
#   XLA_FFI_Extension_Base* extension_start;
#   int major_version;  // out
#   int minor_version;  // out
# };
class XLA_FFI_Api_Version(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_size_t),
                ("extension_start", ctypes.POINTER(XLA_FFI_Extension_Base)),
                ("major_version", ctypes.c_int),
                ("minor_version", ctypes.c_int)]

# enum XLA_FFI_Handler_TraitsBits {
#   // Calls to FFI handler are safe to trace into the command buffer. It means
#   // that calls to FFI handler always launch exactly the same device operations
#   // (can depend on attribute values) that can be captured and then replayed.
#   XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE = 1u << 0,
# };
class XLA_FFI_Handler_TraitsBits(enum.IntEnum):
    COMMAND_BUFFER_COMPATIBLE = 1 << 0

# struct XLA_FFI_Metadata {
#   size_t struct_size;
#   XLA_FFI_Api_Version api_version;
#   XLA_FFI_Handler_Traits traits;
# };
class XLA_FFI_Metadata(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_size_t),
                ("api_version", XLA_FFI_Api_Version), # XLA_FFI_Extension_Type
                ("traits", ctypes.c_uint32) # XLA_FFI_Handler_Traits
    ]

# struct XLA_FFI_Metadata_Extension {
#   XLA_FFI_Extension_Base extension_base;
#   XLA_FFI_Metadata* metadata;
# };
class XLA_FFI_Metadata_Extension(ctypes.Structure):
    _fields_ = [("extension_base", XLA_FFI_Extension_Base),
                ("metadata", ctypes.POINTER(XLA_FFI_Metadata))]

# typedef enum {
#   XLA_FFI_Error_Code_OK = 0,
#   XLA_FFI_Error_Code_CANCELLED = 1,
#   XLA_FFI_Error_Code_UNKNOWN = 2,
#   XLA_FFI_Error_Code_INVALID_ARGUMENT = 3,
#   XLA_FFI_Error_Code_DEADLINE_EXCEEDED = 4,
#   XLA_FFI_Error_Code_NOT_FOUND = 5,
#   XLA_FFI_Error_Code_ALREADY_EXISTS = 6,
#   XLA_FFI_Error_Code_PERMISSION_DENIED = 7,
#   XLA_FFI_Error_Code_RESOURCE_EXHAUSTED = 8,
#   XLA_FFI_Error_Code_FAILED_PRECONDITION = 9,
#   XLA_FFI_Error_Code_ABORTED = 10,
#   XLA_FFI_Error_Code_OUT_OF_RANGE = 11,
#   XLA_FFI_Error_Code_UNIMPLEMENTED = 12,
#   XLA_FFI_Error_Code_INTERNAL = 13,
#   XLA_FFI_Error_Code_UNAVAILABLE = 14,
#   XLA_FFI_Error_Code_DATA_LOSS = 15,
#   XLA_FFI_Error_Code_UNAUTHENTICATED = 16
# } XLA_FFI_Error_Code;
class XLA_FFI_Error_Code(enum.IntEnum):
    OK = 0
    CANCELLED = 1
    UNKNOWN = 2
    INVALID_ARGUMENT = 3
    DEADLINE_EXCEEDED = 4
    NOT_FOUND = 5
    ALREADY_EXISTS = 6
    PERMISSION_DENIED = 7
    RESOURCE_EXHAUSTED = 8
    FAILED_PRECONDITION = 9
    ABORTED = 10
    OUT_OF_RANGE = 11
    UNIMPLEMENTED = 12
    INTERNAL = 13
    UNAVAILABLE = 14
    DATA_LOSS = 15
    UNAUTHENTICATED = 16

# struct XLA_FFI_Error_Create_Args {
#   size_t struct_size;
#   XLA_FFI_Extension_Base* extension_start;
#   const char* message;
#   XLA_FFI_Error_Code errc;
# };
class XLA_FFI_Error_Create_Args(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_size_t),
                ("extension_start", ctypes.POINTER(XLA_FFI_Extension_Base)),
                ("message", ctypes.c_char_p),
                ("errc", ctypes.c_int)] # XLA_FFI_Error_Code

XLA_FFI_Error_Create = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.POINTER(XLA_FFI_Error_Create_Args))

# struct XLA_FFI_Stream_Get_Args {
#   size_t struct_size;
#   XLA_FFI_Extension_Base* extension_start;
#   XLA_FFI_ExecutionContext* ctx;
#   void* stream;  // out
# };
class XLA_FFI_Stream_Get_Args(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_size_t),
                ("extension_start", ctypes.POINTER(XLA_FFI_Extension_Base)),
                ("ctx", ctypes.c_void_p), # XLA_FFI_ExecutionContext*
                ("stream", ctypes.c_void_p)] # // out

XLA_FFI_Stream_Get = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.POINTER(XLA_FFI_Stream_Get_Args))

# struct XLA_FFI_Api {
#   size_t struct_size;
#   XLA_FFI_Extension_Base* extension_start;
#
#   XLA_FFI_Api_Version api_version;
#   XLA_FFI_InternalApi* internal_api;
#
#   _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_Error_Create);
#   _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_Error_GetMessage);
#   _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_Error_Destroy);
#   _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_Handler_Register);
#   _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_Stream_Get);
#   _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_TypeId_Register);
#   _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_ExecutionContext_Get);
#   _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_State_Set);
#   _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_State_Get);
#   _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_DeviceMemory_Allocate);
#   _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_DeviceMemory_Free);
#   _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_ThreadPool_Schedule);
#   _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_ThreadPool_NumThreads);
#   _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_Future_Create);
#   _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_Future_SetAvailable);
#   _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_Future_SetError);
# };
class XLA_FFI_Api(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_size_t),
                ("extension_start", ctypes.POINTER(XLA_FFI_Extension_Base)),
                ("api_version", XLA_FFI_Api_Version),
                ("internal_api", ctypes.c_void_p), # XLA_FFI_InternalApi*

                ("XLA_FFI_Error_Create", XLA_FFI_Error_Create), # XLA_FFI_Error_Create
                ("XLA_FFI_Error_GetMessage", ctypes.c_void_p), # XLA_FFI_Error_GetMessage
                ("XLA_FFI_Error_Destroy", ctypes.c_void_p), # XLA_FFI_Error_Destroy
                ("XLA_FFI_Handler_Register", ctypes.c_void_p), # XLA_FFI_Handler_Register
                ("XLA_FFI_Stream_Get", XLA_FFI_Stream_Get), # XLA_FFI_Stream_Get
                ("XLA_FFI_TypeId_Register", ctypes.c_void_p), # XLA_FFI_TypeId_Register
                ("XLA_FFI_ExecutionContext_Get", ctypes.c_void_p), # XLA_FFI_ExecutionContext_Get
                ("XLA_FFI_State_Set", ctypes.c_void_p), # XLA_FFI_State_Set
                ("XLA_FFI_State_Get", ctypes.c_void_p), # XLA_FFI_State_Get
                ("XLA_FFI_DeviceMemory_Allocate", ctypes.c_void_p), # XLA_FFI_DeviceMemory_Allocate
                ("XLA_FFI_DeviceMemory_Free", ctypes.c_void_p), # XLA_FFI_DeviceMemory_Free
                ("XLA_FFI_ThreadPool_Schedule", ctypes.c_void_p), # XLA_FFI_ThreadPool_Schedule
                ("XLA_FFI_ThreadPool_NumThreads", ctypes.c_void_p), # XLA_FFI_ThreadPool_NumThreads
                ("XLA_FFI_Future_Create", ctypes.c_void_p), # XLA_FFI_Future_Create
                ("XLA_FFI_Future_SetAvailable", ctypes.c_void_p), # XLA_FFI_Future_SetAvailable
                ("XLA_FFI_Future_SetError", ctypes.c_void_p) # XLA_FFI_Future_SetError
    ]

# struct XLA_FFI_CallFrame {
#   size_t struct_size;
#   XLA_FFI_Extension_Base* extension_start;
#   const XLA_FFI_Api* api;
#   XLA_FFI_ExecutionContext* ctx;
#   XLA_FFI_ExecutionStage stage;
#   XLA_FFI_Args args;
#   XLA_FFI_Rets rets;
#   XLA_FFI_Attrs attrs;
#
#   // XLA FFI handler implementation can use `future` to signal a result of
#   // asynchronous computation to the XLA runtime. XLA runtime will keep all
#   // arguments, results and attributes alive until `future` is completed.
#   XLA_FFI_Future* future;  // out
# };
class XLA_FFI_CallFrame(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_size_t),
                ("extension_start", ctypes.POINTER(XLA_FFI_Extension_Base)),
                ("api", ctypes.POINTER(XLA_FFI_Api)),
                ("ctx", ctypes.c_void_p), # XLA_FFI_ExecutionContext*
                ("stage", ctypes.c_int), # XLA_FFI_ExecutionStage
                ("args", XLA_FFI_Args),
                ("rets", XLA_FFI_Rets),
                ("attrs", XLA_FFI_Attrs),
                ("future", ctypes.c_void_p) # XLA_FFI_Future* // out 
    ]

#######################################################################
# End of ctypes structures
#######################################################################


# Holder for the custom callback to keep it alive.
ffi_callback = None
xla_device_to_warp_device = None
registered_kernels = [None]
registered_kernel_to_id = {}

def jax_kernel_call(kernel_id, wp_kernel, *args):
    x_shape = args[0].shape
    size = math.prod(x_shape)

    descriptor = "{0}|{1}".format(kernel_id, size)

    call = jax.ffi.ffi_call(
        "warp_handler",
        jax.ShapeDtypeStruct(x_shape, jax.numpy.float32),
        vmap_method="broadcast_all")
    return call(*args, desc = descriptor)

def jax_kernel(wp_kernel):
    if ffi_callback == None:
        register_warp_ffi_callback()
    id = None
    if not wp_kernel in registered_kernel_to_id:
        id = len(registered_kernels)
        registered_kernels.append(wp_kernel)
        registered_kernel_to_id[wp_kernel] = id
    else:
        id = registered_kernel_to_id[wp_kernel]

    return lambda *args: jax_kernel_call(id, wp_kernel, *args)

########################################################################
# Helpers for translating between ctypes and python types

# XLA_FFI_ByteSpan to string
def string_from_bytespan_ptr(span):
    len = span.contents.len
    chars = ctypes.cast(span.contents.ptr, ctypes.POINTER(ctypes.c_char * len))
    return chars.contents.value.decode("utf-8")

# error-string to XLA_FFI_Error
def create_invalid_argument_ffi_error(api, message):
    create_args = XLA_FFI_Error_Create_Args(
        ctypes.sizeof(XLA_FFI_Error_Create_Args),
        ctypes.POINTER(XLA_FFI_Extension_Base)(),
        ctypes.c_char_p(message.encode('utf-8')),
        XLA_FFI_Error_Code.INVALID_ARGUMENT
    )
    return api.contents.XLA_FFI_Error_Create(create_args)

import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# Extract CUDA stream from XLA_FFI_CallFrame.
def get_stream_from_callframe(call_frame):
    api = call_frame.contents.api
    get_stream_args = XLA_FFI_Stream_Get_Args(
        ctypes.sizeof(XLA_FFI_Stream_Get_Args),
        ctypes.POINTER(XLA_FFI_Extension_Base)(),
        call_frame.contents.ctx,
        None
    )
    api.contents.XLA_FFI_Stream_Get(get_stream_args)
    # TODO check result
    return get_stream_args.stream

def warp_ffi_callback(call_frame):
    # TODO Try-catch around the body and return XLA_FFI_Error on error.
    extension = call_frame.contents.extension_start
    # On the first call, XLA runtime will query the API version and traits
    # metadata using the |extension| field. Let us respond to that query
    # if the metadata extension is present.
    if extension:
        # Try to set the version metadata.
        if extension.contents.type == XLA_FFI_Extension_Type.Metadata:
            metadata_ext = ctypes.cast(extension, ctypes.POINTER(XLA_FFI_Metadata_Extension))
            metadata_ext.contents.metadata.contents.api_version.major_version = 0
            metadata_ext.contents.metadata.contents.api_version.minor_version = 1
            # Turn on CUDA graphs for this handler.
            metadata_ext.contents.metadata.contents.traits = XLA_FFI_Handler_TraitsBits.COMMAND_BUFFER_COMPATIBLE
            return None
    
    if (call_frame.contents.attrs.size != 1 or 
            call_frame.contents.attrs.types[0] != XLA_FFI_AttrType.STRING):
        return create_invalid_argument_ffi_error(
            call_frame.contents.api,
            "Internal error, expected one attribute ('desc')")

    attr_name = string_from_bytespan_ptr(call_frame.contents.attrs.names[0])
    if attr_name != "desc":
        return create_invalid_argument_ffi_error(
            call_frame.contents.api,
            "Internal error, expected 'desc' attribute")

    attr_value = ctypes.cast(call_frame.contents.attrs.attrs[0], ctypes.POINTER(XLA_FFI_ByteSpan))
    descriptor = string_from_bytespan_ptr(attr_value)

    [kernel_id_str, length_str] = descriptor.split('|')
    kernel = registered_kernels[int(kernel_id_str)]
    # Currently we only pass one number that is used for input length,
    # output length and iteration count. 
    length = int(length_str)

    xla_stream = get_stream_from_callframe(call_frame)

    input_count = call_frame.contents.args.size
    if input_count != 1:
        return create_invalid_argument_ffi_error(
            call_frame.contents.api,
            "Only one input supported in the PoC")
    inputs = ctypes.cast(call_frame.contents.args.args, ctypes.POINTER(ctypes.POINTER(XLA_FFI_Buffer)))
    input = inputs.contents[0].data
    
    output_count = call_frame.contents.rets.size
    if output_count != 1:
        return create_invalid_argument_ffi_error(
            call_frame.contents.api,
            "Only one output supported in the PoC")
    outputs = ctypes.cast(call_frame.contents.rets.rets, ctypes.POINTER(ctypes.POINTER(XLA_FFI_Buffer)))
    output = outputs.contents[0].data

    stream = wp.Stream(cuda_stream=ctypes.c_void_p(xla_stream))

    # TODO: Generate shapes and types from descriptor.
    # TODO: Allow arbitrary number of inputs/outputs. (Currently we only 
    # allow one float array input and one float array output.)
    a0 = wp.array(ptr=input, dtype=float, shape=length, owner=False, copy=False)
    a1 = wp.array(ptr=output, dtype=float, shape=length, owner=False, copy=False)
    # wp.capture_begin(stream=stream, force_module_load=False, external=True)
    wp.launch(kernel, [length], [a0, a1], stream=stream)
    # wp.capture_end()
    return None


def register_warp_ffi_callback():
    global ffi_callback

    # This is necessary to avoid initializing warp device context
    # during warp capture (warp device context initialization causes
    # CUDA graph capture failures).
    for device in jax.local_devices():
        wp.jax.device_from_jax(jax.local_devices()[0]).context

    FFI_CCALLFUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(XLA_FFI_CallFrame))
    ffi_callback = FFI_CCALLFUNC(warp_ffi_callback)
    ffi_ccall_address = ctypes.cast(ffi_callback, ctypes.c_void_p)
    ffi_capsule = jax.ffi.pycapsule(ffi_ccall_address.value)
    jax.ffi.register_ffi_target("warp_handler", ffi_capsule, platform="CUDA")

wp.init()

@wp.kernel
def triple(input: wp.array(dtype=float),
           output: wp.array(dtype=float)):
    tid = wp.tid()
    output[tid] = 3.0 * input[tid]

jax_triple = jax_kernel(triple)

@jax.jit
def f():
    x = jp.arange(0, 256, dtype = jp.float32)
    for _ in range(10):
        x = jax_triple(x)
    return x

print(f())

import time
start = time.time()
for i in range(100):
    f().block_until_ready()
end = time.time()
print("Elapsed: ", end - start)
