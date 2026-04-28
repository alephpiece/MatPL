from contextlib import nullcontext


class ProfilingConfig(object):
    def __init__(self, raw=False):
        if raw is None:
            raw = False
        if isinstance(raw, bool):
            self.enabled = raw
            raw = {}
        elif isinstance(raw, dict):
            self.enabled = raw.get("enabled", True)
        else:
            raise TypeError("profiling must be a bool or object")
        self._reject_unknown(
            raw,
            {
                "enabled", "activities", "record_shapes", "profile_memory", "with_stack",
                "with_flops", "schedule", "tensorboard_trace", "summary",
            },
            "profiling",
        )

        self.activities = self._parse_activities(raw.get("activities", ["cpu", "gpu"]))
        self.record_shapes = self._get_bool(raw, "record_shapes", True)
        self.profile_memory = self._get_bool(raw, "profile_memory", False)
        self.with_stack = self._get_bool(raw, "with_stack", False)
        self.with_flops = self._get_bool(raw, "with_flops", False)

        self.schedule = self._parse_schedule(raw.get("schedule", {}))
        self.tensorboard_trace = self._parse_tensorboard_trace(raw)
        self.summary = self._parse_summary(raw.get("summary", {}))
        if self.enabled and not any(
            target["enabled"] for target in [self.tensorboard_trace, self.summary]
        ):
            raise ValueError("profiling enabled but no export target is enabled")

    @staticmethod
    def _reject_unknown(raw, allowed, context):
        unknown = set(raw.keys()) - set(allowed)
        if unknown:
            raise ValueError("{} has unknown fields: {}".format(context, ", ".join(sorted(unknown))))

    @staticmethod
    def _get_bool(raw, key, default):
        value = raw.get(key, default)
        if not isinstance(value, bool):
            raise TypeError("profiling.{} must be a bool".format(key))
        return value

    @staticmethod
    def _parse_activities(value):
        if not isinstance(value, list):
            raise TypeError("profiling.activities must be a list")
        activities = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("profiling.activities entries must be strings")
            item = item.lower()
            if item == "cuda":
                item = "gpu"
            if item not in ["cpu", "gpu"]:
                raise ValueError("profiling.activities only supports 'cpu' and 'gpu'")
            if item not in activities:
                activities.append(item)
        if not activities:
            raise ValueError("profiling.activities must not be empty")
        return activities

    @staticmethod
    def _parse_schedule(value):
        if not isinstance(value, dict):
            raise TypeError("profiling.schedule must be an object")
        ProfilingConfig._reject_unknown(
            value,
            {"wait", "warmup", "active", "repeat", "skip_first"},
            "profiling.schedule",
        )
        schedule = {
            "wait": value.get("wait", 1),
            "warmup": value.get("warmup", 1),
            "active": value.get("active", 3),
            "repeat": value.get("repeat", 1),
            "skip_first": value.get("skip_first", 0),
        }
        for key, val in schedule.items():
            if not isinstance(val, int) or val < 0:
                raise ValueError("profiling.schedule.{} must be a non-negative integer".format(key))
        if schedule["active"] <= 0:
            raise ValueError("profiling.schedule.active must be greater than zero")
        return schedule

    @staticmethod
    def _has_export_target(raw):
        return any(key in raw for key in ["tensorboard_trace", "summary"])

    def _parse_tensorboard_trace(self, raw):
        default_enabled = not self._has_export_target(raw)
        value = raw.get("tensorboard_trace", {})
        if not isinstance(value, dict):
            raise TypeError("profiling.tensorboard_trace must be an object")
        ProfilingConfig._reject_unknown(
            value,
            {"enabled", "dir_name", "worker_name", "use_gzip"},
            "profiling.tensorboard_trace",
        )
        config = {
            "enabled": value.get("enabled", default_enabled),
            "dir_name": value.get("dir_name", "profiler/tensorboard"),
            "worker_name": value.get("worker_name", None),
            "use_gzip": value.get("use_gzip", False),
        }
        if not isinstance(config["enabled"], bool):
            raise TypeError("profiling.tensorboard_trace.enabled must be a bool")
        if not isinstance(config["dir_name"], str):
            raise TypeError("profiling.tensorboard_trace.dir_name must be a string")
        if config["worker_name"] is not None and not isinstance(config["worker_name"], str):
            raise TypeError("profiling.tensorboard_trace.worker_name must be a string or null")
        if not isinstance(config["use_gzip"], bool):
            raise TypeError("profiling.tensorboard_trace.use_gzip must be a bool")
        return config

    @staticmethod
    def _parse_summary(value):
        if not isinstance(value, dict):
            raise TypeError("profiling.summary must be an object")
        ProfilingConfig._reject_unknown(
            value,
            {"enabled", "sort_by", "row_limit"},
            "profiling.summary",
        )
        sort_by = value.get("sort_by", "gpu_time_total")
        if sort_by == "gpu_time_total":
            sort_by = "cuda_time_total"
        elif sort_by == "self_gpu_time_total":
            sort_by = "self_cuda_time_total"
        config = {
            "enabled": value.get("enabled", False),
            "sort_by": sort_by,
            "row_limit": value.get("row_limit", 30),
        }
        if not isinstance(config["enabled"], bool):
            raise TypeError("profiling.summary.enabled must be a bool")
        if not isinstance(config["sort_by"], str):
            raise TypeError("profiling.summary.sort_by must be a string")
        if not isinstance(config["row_limit"], int) or config["row_limit"] < 0:
            raise ValueError("profiling.summary.row_limit must be a non-negative integer")
        return config

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "activities": self.activities,
            "record_shapes": self.record_shapes,
            "profile_memory": self.profile_memory,
            "with_stack": self.with_stack,
            "with_flops": self.with_flops,
            "schedule": dict(self.schedule),
            "tensorboard_trace": dict(self.tensorboard_trace),
            "summary": dict(self.summary),
        }

    def __bool__(self):
        return self.enabled


class MatPLProfiler(object):
    def __init__(self, config, trace_name="train"):
        if isinstance(config, ProfilingConfig):
            self.config = config
        else:
            self.config = ProfilingConfig(config)
        self.trace_name = trace_name
        self.profiler = None
        self.record_function = None

    @property
    def enabled(self):
        return self.config.enabled

    def __enter__(self):
        if not self.enabled:
            return self
        import torch
        from torch.profiler import ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler

        activities = []
        if "cpu" in self.config.activities:
            activities.append(ProfilerActivity.CPU)
        if "gpu" in self.config.activities and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        if not activities:
            activities.append(ProfilerActivity.CPU)

        callbacks = []
        if self.config.tensorboard_trace["enabled"]:
            tb = self.config.tensorboard_trace
            callbacks.append(
                tensorboard_trace_handler(
                    tb["dir_name"],
                    worker_name=tb["worker_name"],
                    use_gzip=tb["use_gzip"],
                )
            )
        if self.config.summary["enabled"]:
            callbacks.append(self._summary_callback)

        def on_trace_ready(prof):
            for callback in callbacks:
                callback(prof)

        self.record_function = record_function
        self.profiler = profile(
            activities=activities,
            schedule=schedule(**self.config.schedule),
            on_trace_ready=on_trace_ready if callbacks else None,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_flops=self.config.with_flops,
        )
        self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.profiler is not None:
            return self.profiler.__exit__(exc_type, exc, tb)
        return False

    def start(self):
        return self.__enter__()

    def stop(self):
        return self.__exit__(None, None, None)

    def record(self, name):
        if self.profiler is None or self.record_function is None:
            return nullcontext()
        return self.record_function(name)

    def step(self):
        if self.profiler is not None:
            self.profiler.step()

    def _summary_callback(self, prof):
        summary = self.config.summary
        sort_by = summary["sort_by"]
        if "cuda" in sort_by:
            try:
                import torch
                if not torch.cuda.is_available():
                    sort_by = "cpu_time_total"
            except Exception:
                sort_by = "cpu_time_total"
        kwargs = {"sort_by": sort_by}
        if summary["row_limit"] > 0:
            kwargs["row_limit"] = summary["row_limit"]
        print(prof.key_averages().table(**kwargs))
