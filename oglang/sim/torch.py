    # def forward(self, model: Model, state_in: State, dt: float) -> State:
    #     """Performs a single integration step forward in time
        
    #     This method inserts a node into the PyTorch computational graph with
    #     references to all model and state tensors such that gradients
    #     can be propagrated back through the simulation step.

    #     Args:

    #         model: Simulation model
    #         state: Simulation state at the start the time-step
    #         dt: The simulation time-step (usually in seconds)

    #     Returns:

    #         The state of the system at the end of the time-step

    #     """


    #     if og.config.no_grad:

    #         # if no gradient required then do inplace update
    #         self._simulate(og.Tape(), model, state_in, state_in, dt)
    #         return state_in

    #     else:

    #         # get list of inputs and outputs for PyTorch tensor tracking            
    #         inputs = [*state_in.flatten(), *model.flatten()]

    #         # allocate new output
    #         state_out = model.state()

    #         # run sim as a PyTorch op
    #         tensors = SimulateFunc.apply(self, model, state_in, state_out, dt, *inputs)

    #         return state_out
            



# # define PyTorch autograd op to wrap simulate func
# class SimulateFunc(torch.autograd.Function):
#     """PyTorch autograd function representing a simulation stpe
    
#     Note:
    
#         This node will be inserted into the computation graph whenever
#         `forward()` is called on an integrator object. It should not be called
#         directly by the user.        
#     """

#     @staticmethod
#     def forward(ctx, integrator, model, state_in, state_out, dt, *tensors):

#         # record launches
#         ctx.og = og.og()
#         ctx.inputs = tensors
#         ctx.outputs = og.to_weak_list(state_out.flatten())

#         # simulate
#         integrator._simulate(ctx.og, model, state_in, state_out, dt)

#         return tuple(state_out.flatten())

#     @staticmethod
#     def backward(ctx, *grads):

#         # ensure grads are contiguous in memory
#         adj_outputs = og.make_contiguous(grads)

#         # register outputs with og
#         outputs = og.to_strong_list(ctx.outputs)        
#         for o in range(len(outputs)):
#             ctx.og.adjoints[outputs[o]] = adj_outputs[o]

#         # replay launches backwards
#         ctx.og.replay()

#         # find adjoint of inputs
#         adj_inputs = []
#         for i in ctx.inputs:

#             if i in ctx.og.adjoints:
#                 adj_inputs.append(ctx.og.adjoints[i])
#             else:
#                 adj_inputs.append(None)


#         # free the og
#         ctx.og.reset()

#         # filter grads to replace empty tensors / no grad / constant params with None
#         return (None, None, None, None, None, *og.filter_grads(adj_inputs))
