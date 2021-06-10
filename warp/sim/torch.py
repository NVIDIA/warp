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


    #     if wp.config.no_grad:

    #         # if no gradient required then do inplace update
    #         self._simulate(wp.Tape(), model, state_in, state_in, dt)
    #         return state_in

    #     else:

    #         # get list of inputs and outputs for PyTorch tensor tracking            
    #         inputs = [*state_in.flatten(), *model.flatten()]

    #         # allocate new output
    #         state_out = model.state()

    #         # run sim as a PyTorch op
    #         tensors = SimulateFunc.apply(self, model, state_in, state_out, dt, *inputs)

    #         return state_out
            



# # define PyTorch autwprad op to wrap simulate func
# class SimulateFunc(torch.autwprad.Function):
#     """PyTorch autwprad function representing a simulation stpe
    
#     Note:
    
#         This node will be inserted into the computation graph whenever
#         `forward()` is called on an integrator object. It should not be called
#         directly by the user.        
#     """

#     @staticmethod
#     def forward(ctx, integrator, model, state_in, state_out, dt, *tensors):

#         # record launches
#         ctx.wp = wp.wp()
#         ctx.inputs = tensors
#         ctx.outputs = wp.to_weak_list(state_out.flatten())

#         # simulate
#         integrator._simulate(ctx.wp, model, state_in, state_out, dt)

#         return tuple(state_out.flatten())

#     @staticmethod
#     def backward(ctx, *grads):

#         # ensure grads are contiguous in memory
#         adj_outputs = wp.make_contiguous(grads)

#         # register outputs with wp
#         outputs = wp.to_strong_list(ctx.outputs)        
#         for o in range(len(outputs)):
#             ctx.wp.adjoints[outputs[o]] = adj_outputs[o]

#         # replay launches backwards
#         ctx.wp.replay()

#         # find adjoint of inputs
#         adj_inputs = []
#         for i in ctx.inputs:

#             if i in ctx.wp.adjoints:
#                 adj_inputs.append(ctx.wp.adjoints[i])
#             else:
#                 adj_inputs.append(None)


#         # free the wp
#         ctx.wp.reset()

#         # filter grads to replace empty tensors / no grad / constant params with None
#         return (None, None, None, None, None, *wp.filter_grads(adj_inputs))
