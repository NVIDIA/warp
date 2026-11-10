Fix `wp.Tape.backward()` seeding the adjoints of a struct argument's scalar, vector, matrix, quaternion, and
transformation fields with the argument's values instead of zero, which gave wrong gradients when a kernel or a
function it calls assigns to such a field, for example by updating a struct argument and returning it.
