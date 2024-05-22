from nodes import NODE_CLASS_MAPPINGS
from torch.profiler import ProfilerActivity, profile, record_function

profile_node_ids = ["KSampler", "SamplerCustom"]

for profile_node_id in profile_node_ids:
    profile_node_cls = NODE_CLASS_MAPPINGS[profile_node_id]
    profile_node_function_name = profile_node_cls.FUNCTION

    class ProfileProxy(profile_node_cls):
        FUNCTION = "profile_proxy_exec"
        ORIGIN_FUNCTION = profile_node_function_name

        def profile_proxy_exec(self, *args, **kwargs):
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                with record_function("model_inference"):
                    origin_function = getattr(self, self.ORIGIN_FUNCTION)
                    result = origin_function(*args, **kwargs)

            # Print profiling results
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

            # Export to Chrome trace format
            prof.export_chrome_trace("trace.json")
            return result

    NODE_CLASS_MAPPINGS[profile_node_id] = ProfileProxy
