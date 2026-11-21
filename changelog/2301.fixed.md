Fix component writes such as `s.v[0] = x`, `s.v.y = x`, and `s.m[i, j] += x` to vector, matrix, quaternion, and
transformation fields of a local struct silently producing zero gradients. Assigning `s.t.p` or `s.t.q` for a
transformation field `s.t`, or a slice such as `s.v[1:]`, no longer raises a code generation error.
