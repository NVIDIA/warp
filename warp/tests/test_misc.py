import warp as wp

wp.init()

@wp.kernel
def test_print():

    ubyte_a = wp.uint8(36)
    ubyte_b = wp.uint8(32)

    ushort_a = wp.uint16(36)
    ushort_b = wp.uint16(32)

    uint_a = wp.uint32(36)
    uint_b = wp.uint32(32)

    ulong_a = wp.uint64(36)
    ulong_b = wp.uint64(32)

    byte_a = wp.int8(36)
    byte_b = wp.int8(32)

    int_a = wp.int32(36)
    int_b = wp.int32(32)

    short_a = wp.int16(36)
    short_b = wp.int16(32)

    long_a = wp.int64(36)
    long_b = wp.int64(32)


    print(ubyte_a + ubyte_b)
    print(ushort_a + ushort_b)
    print(uint_a + uint_b)
    print(ulong_a + ulong_b)
    
    print(byte_a + byte_b)
    print(short_a + short_b)
    print(int_a + int_b)
    print(long_a + long_b)


wp.launch(test_print, dim=1, inputs=[])
wp.synchronize()

    

@wp.kernel
def test_vec(a: wp.vec(length=5, dtype=wp.int16)):

    #byte_a = wp.uint8(36)
    #byte_b = wp.uint8(32)

    #print(byte_a + byte_b)

    b = a * wp.int16(-2)
    print(b)



v3 = wp.vec3(1.0, 2.0, 3.0)
vec5 = wp.vec(length=5, dtype=wp.int16)
v5 = vec5(1, 2, 3, 4, 5)

wp.launch(test_vec, dim=1, inputs=[v5])
wp.synchronize()
