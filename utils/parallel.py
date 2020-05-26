from torch.nn.parallel import DataParallel, DistributedDataParallel
from itertools import chain


class DataParallelMore(DataParallel):


    def forward_teach(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module.forward_teach(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)


    def set_choice(self, choice):
        self.module.set_choice(choice)

    @property
    def random_shuffle(self):
        return self.module.random_shuffle

    @property
    def random_choice(self):
        return self.module.random_choice

    def forward_distill(self, *inputs, **kwargs):
        assert False
        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        stack_inps = []
        for inp in inputs:
            inp = self.scatter(inp, kwargs, self.device_ids)
            stack_inps.append(inp)
        if len(self.device_ids) == 1:
            return self.module.forward_distill(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, stack_inps, kwargs)
        return self.gather(outputs, self.output_device)


def is_parallel(model):
    return True if isinstance(
        model, (DistributedDataParallel, DataParallel, DataParallelMore)) else False


def get_state_dict(model):
    if is_parallel(model):
        return model.module.state_dict()
    else:
        return model.state_dict()
