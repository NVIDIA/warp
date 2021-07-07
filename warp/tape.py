import warp as wp

class Tape:

    def __init__(self):

        self.adjoints = {}
        self.launches = []

    # launch a kernel and record on the tape
    def launch(self, kernel, dim, inputs, outputs, device):
        
        wp.launch(
            kernel=kernel, 
            dim=dim, 
            inputs=inputs, 
            outputs=outputs, 
            device=device)

        self.launches.append([kernel, dim, inputs, outputs, device])

    # adj_outputs is a mapping from output tensor -> adjoint of the output
    # after running backward the adjoints of tensors may be retrieved by:
    #
    #  adj_tensor = tape.adjoints[tensor]
    #
    def backward(self, adj_user: dict):

        # insert user specified adjoints (e.g.: from loss function inputs) into our lookup
        for a in adj_user:
            self.adjoints.update(adj_user)

        # run launches backwards
        for launch in reversed(self.launches):

            kernel = launch[0]
            dim = launch[1]
            inputs = launch[2]
            outputs = launch[3]
            device = launch[4]

            adj_inputs = []
            adj_outputs = []

            # lookup adjoint inputs
            for a in inputs:
                adj_inputs.append(self.get_adjoint(a))

            # lookup adjoint outputs, todo: only allocate outputs if necessary
            for a in outputs:
                adj_outputs.append(self.get_adjoint(a))

            wp.launch(
                kernel=kernel, 
                dim=dim, 
                inputs=inputs, 
                outputs=outputs,
                adj_inputs=adj_inputs,
                adj_outputs=adj_outputs,
                device=device,
                adjoint=True)

    # returns the adjoint version of a tensor used in the computation
    def get_adjoint(self, a):
        
        if (a in self.adjoints):
            return self.adjoints[a]
        else:

            if (wp.type_is_int(a.dtype) or a.requires_grad == False):
                return None
            else:
                # alloc adjoint for variable
                adj = wp.zeros_like(a)
                self.adjoints[a] = adj
                return adj

    def reset(self):

        self.adjoints = {}
        self.launches = []
        
   