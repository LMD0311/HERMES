from mmcv.runner.hooks.hook import HOOKS, Hook
from projects.mmdet3d_plugin.models.utils import run_time
import torch


@HOOKS.register_module()
class TransferWeight(Hook):

    def __init__(self, every_n_inters=1):
        self.every_n_inters = every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            runner.eval_model.load_state_dict(runner.model.state_dict())


# @HOOKS.register_module()
# class ToFloat32Hook(Hook):
#     def _set_module_dtype(self, runner, module_path_str: str, target_dtype: torch.dtype):
#         """
#         Sets the data type of the module specified by the path.
#         module_path_str: Dot-separated module path string, e.g., 'pts_bbox_head.llm'.
#         target_dtype: The target torch.dtype.
#         """
#         if not hasattr(runner, 'model') or not hasattr(runner.model, 'module'):
#             print(f"Warning: runner.model.module not found. Cannot process module path '{module_path_str}'.")
#             return
#
#         top_module = runner.model.module
#         parts = module_path_str.split('.')
#         current_obj = top_module
#
#         # Traverse the path to find the target module
#         for i, part_name in enumerate(parts):
#             if not hasattr(current_obj, part_name):
#                 # Construct the path traversed so far for a clearer error message
#                 path_so_far_str = '.'.join(parts[:i])
#                 parent_module_display_name = type(top_module).__name__ if not path_so_far_str else path_so_far_str
#                 print(
#                     f"Warning: When trying to access '{module_path_str}', attribute '{part_name}' was not found in '{parent_module_display_name}'.")
#                 return  # Module path is invalid
#
#             current_obj = getattr(current_obj, part_name)
#
#             # Whether it's an intermediate part of the path or the final target, we expect it to be a torch.nn.Module
#             if not isinstance(current_obj, torch.nn.Module):
#                 current_path_str = '.'.join(parts[:i + 1])
#                 print(
#                     f"Warning: Attribute '{current_path_str}' in path '{module_path_str}' resolved to a non-nn.Module object (type: {type(current_obj).__name__}). Cannot convert.")
#                 return  # Target or intermediate path is not a module
#
#         # If the entire path was successfully traversed, current_obj is the module we want to convert
#         module_to_convert = current_obj
#
#         # Double-check that module_to_convert is an nn.Module (theoretically guaranteed by previous checks)
#         if isinstance(module_to_convert, torch.nn.Module):
#             module_to_convert.to(target_dtype)
#             module_to_convert.fp16_enabled = False
#             # print(f"Successfully set: Module '{module_path_str}' data type to {target_dtype}")
#         else:
#             # This case should theoretically be caught by the checks within the loop
#             print(
#                 f"Error: Target '{module_path_str}' is not a torch.nn.Module (actual type: {type(module_to_convert).__name__}). Cannot convert.")
#
#     def before_run(self, runner):
#         # Convert specified modules to their target data types
#         self._set_module_dtype(runner, 'img_backbone', torch.float32)
#         self._set_module_dtype(runner, 'input_proj', torch.float32)
#         self._set_module_dtype(runner, 'img_neck', torch.float32)
#         self._set_module_dtype(runner, 'pts_bbox_head', torch.float32)
#         self._set_module_dtype(runner, 'pts_bbox_head.llm', torch.bfloat16)
#         self._set_module_dtype(runner, 'pts_bbox_head.llm.in_mlp', torch.float32)
#         self._set_module_dtype(runner, 'pts_bbox_head.llm.out_mlp', torch.float32)
#         torch.cuda.empty_cache()
#
#     def after_train_iter(self, runner):
#         # Convert specified modules to their target data types
#         self._set_module_dtype(runner, 'img_backbone', torch.float32)
#         self._set_module_dtype(runner, 'input_proj', torch.float32)
#         self._set_module_dtype(runner, 'img_neck', torch.float32)
#         self._set_module_dtype(runner, 'pts_bbox_head', torch.float32)
#         self._set_module_dtype(runner, 'pts_bbox_head.llm', torch.bfloat16)
#         self._set_module_dtype(runner, 'pts_bbox_head.llm.in_mlp', torch.float32)
#         self._set_module_dtype(runner, 'pts_bbox_head.llm.out_mlp', torch.float32)

@HOOKS.register_module()
class ToFloat32Hook(Hook):
    def __init__(self):
        # Define module-dtype mapping configuration
        self.module_dtype_config = {
            'img_backbone': torch.float32,
            'input_proj': torch.float32,
            'img_neck': torch.float32,
            'pts_bbox_head': torch.float32,
            'pts_bbox_head.llm': torch.bfloat16,  # Override parent dtype
            'pts_bbox_head.llm.in_mlp': torch.float32,  # Override parent dtype
            'pts_bbox_head.llm.out_mlp': torch.float32,  # Override parent dtype
        }
        # Sort configurations by path length to ensure parent modules are processed before children
        self.sorted_module_paths = sorted(self.module_dtype_config.keys(),
                                         key=lambda x: (len(x.split('.')), x))

    def _set_module_dtype(self, runner, module_path_str: str, target_dtype: torch.dtype):
        """
        Sets the data type of the module specified by the path.
        module_path_str: Dot-separated module path string, e.g., 'pts_bbox_head.llm'.
        target_dtype: The target torch.dtype.
        """
        if not hasattr(runner, 'model') or not hasattr(runner.model, 'module'):
            print(f"Warning: runner.model.module not found. Cannot process module path '{module_path_str}'.")
            return

        top_module = runner.model.module
        parts = module_path_str.split('.')
        current_obj = top_module

        # Traverse the path to find the target module
        for i, part_name in enumerate(parts):
            if not hasattr(current_obj, part_name):
                # Construct the path traversed so far for a clearer error message
                path_so_far_str = '.'.join(parts[:i])
                parent_module_display_name = type(top_module).__name__ if not path_so_far_str else path_so_far_str
                print(
                    f"Warning: When trying to access '{module_path_str}', attribute '{part_name}' was not found in '{parent_module_display_name}'.")
                return  # Module path is invalid

            current_obj = getattr(current_obj, part_name)

            # Whether it's an intermediate part of the path or the final target, we expect it to be a torch.nn.Module
            if not isinstance(current_obj, torch.nn.Module):
                current_path_str = '.'.join(parts[:i + 1])
                print(
                    f"Warning: Attribute '{current_path_str}' in path '{module_path_str}' resolved to a non-nn.Module object (type: {type(current_obj).__name__}). Cannot convert.")
                return  # Target or intermediate path is not a module

        # If the entire path was successfully traversed, current_obj is the module we want to convert
        module_to_convert = current_obj

        # Double-check that module_to_convert is an nn.Module
        if isinstance(module_to_convert, torch.nn.Module):
            module_to_convert.to(target_dtype)
            module_to_convert.fp16_enabled = False
            # print(f"Successfully set: Module '{module_path_str}' data type to {target_dtype}")
        else:
            print(
                f"Error: Target '{module_path_str}' is not a torch.nn.Module (actual type: {type(module_to_convert).__name__}). Cannot convert.")

    def _apply_dtype_config(self, runner):
        """Apply all data type configurations in the correct order"""
        for module_path in self.sorted_module_paths:
            target_dtype = self.module_dtype_config[module_path]
            self._set_module_dtype(runner, module_path, target_dtype)

    def before_run(self, runner):
        """Apply data type configurations before the run starts"""
        self._apply_dtype_config(runner)
        torch.cuda.empty_cache()

    def after_train_iter(self, runner):
        """Reapply data type configurations after each training iteration"""
        self._apply_dtype_config(runner)


@HOOKS.register_module()
class Stage3ToFloat32Hook(Hook):
    def __init__(self):
        # Define module-dtype mapping configuration
        self.module_dtype_config = {
            'img_backbone': torch.float32,
            'input_proj': torch.float32,
            'img_neck': torch.float32,
            'pts_bbox_head': torch.float32,
            'pts_bbox_head.llm': torch.bfloat16,  # Override parent dtype
            'pts_bbox_head.llm.out_mlp': torch.float32,  # Override parent dtype
        }
        # Sort configurations by path length to ensure parent modules are processed before children
        self.sorted_module_paths = sorted(self.module_dtype_config.keys(),
                                         key=lambda x: (len(x.split('.')), x))

    def _set_module_dtype(self, runner, module_path_str: str, target_dtype: torch.dtype):
        """
        Sets the data type of the module specified by the path.
        module_path_str: Dot-separated module path string, e.g., 'pts_bbox_head.llm'.
        target_dtype: The target torch.dtype.
        """
        if not hasattr(runner, 'model') or not hasattr(runner.model, 'module'):
            print(f"Warning: runner.model.module not found. Cannot process module path '{module_path_str}'.")
            return

        top_module = runner.model.module
        parts = module_path_str.split('.')
        current_obj = top_module

        # Traverse the path to find the target module
        for i, part_name in enumerate(parts):
            if not hasattr(current_obj, part_name):
                # Construct the path traversed so far for a clearer error message
                path_so_far_str = '.'.join(parts[:i])
                parent_module_display_name = type(top_module).__name__ if not path_so_far_str else path_so_far_str
                print(
                    f"Warning: When trying to access '{module_path_str}', attribute '{part_name}' was not found in '{parent_module_display_name}'.")
                return  # Module path is invalid

            current_obj = getattr(current_obj, part_name)

            # Whether it's an intermediate part of the path or the final target, we expect it to be a torch.nn.Module
            if not isinstance(current_obj, torch.nn.Module):
                current_path_str = '.'.join(parts[:i + 1])
                print(
                    f"Warning: Attribute '{current_path_str}' in path '{module_path_str}' resolved to a non-nn.Module object (type: {type(current_obj).__name__}). Cannot convert.")
                return  # Target or intermediate path is not a module

        # If the entire path was successfully traversed, current_obj is the module we want to convert
        module_to_convert = current_obj

        # Double-check that module_to_convert is an nn.Module
        if isinstance(module_to_convert, torch.nn.Module):
            module_to_convert.to(target_dtype)
            module_to_convert.fp16_enabled = False
            # print(f"Successfully set: Module '{module_path_str}' data type to {target_dtype}")
        else:
            print(
                f"Error: Target '{module_path_str}' is not a torch.nn.Module (actual type: {type(module_to_convert).__name__}). Cannot convert.")

    def _apply_dtype_config(self, runner):
        """Apply all data type configurations in the correct order"""
        for module_path in self.sorted_module_paths:
            target_dtype = self.module_dtype_config[module_path]
            self._set_module_dtype(runner, module_path, target_dtype)

    def before_run(self, runner):
        """Apply data type configurations before the run starts"""
        self._apply_dtype_config(runner)
        torch.cuda.empty_cache()

    def after_train_iter(self, runner):
        """Reapply data type configurations after each training iteration"""
        self._apply_dtype_config(runner)


@HOOKS.register_module()
class EmptyCacheIterHook(Hook):
    def __init__(self, every_n_iters=1):
        self.every_n_iters = every_n_iters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_iters):
            torch.cuda.empty_cache()