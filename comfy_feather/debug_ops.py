from comfy.ops import manual_cast

from .ops import check_tensor, stat_tensor


class DebugOps(manual_cast):
    class Linear(manual_cast.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prefix = None

        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
            super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
            self.prefix = prefix

        def forward(self, x, *args, **kwargs):
            check_tensor(x, self.prefix + "x in forward")
            stat_tensor(x, self.prefix + "x")
            y = super().forward(x, *args, **kwargs)
            check_tensor(y, self.prefix + "y in forward")
            stat_tensor(y, self.prefix + "y")
            return y
