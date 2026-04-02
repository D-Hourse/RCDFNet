from pathlib import Path
import subprocess
import sys

from setuptools import Command, find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).resolve().parent


def run_build_ops_once(cmd_obj):
    """Avoid running expensive op builds multiple times in one install invocation."""
    dist = cmd_obj.distribution
    if getattr(dist, '_rcdbf_ops_built', False):
        cmd_obj.announce('[SKIP] build_rcdfnet_ops already executed', level=2)
        return
    cmd_obj.run_command('build_rcdfnet_ops')
    setattr(dist, '_rcdbf_ops_built', True)


class BuildRcdbfOps(Command):
    """Build CUDA/C++ extensions under rcdfnet/ops and ingroup_indices."""

    description = 'build third-party CUDA ops in rcdfnet/ops'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        def build_ext_in_dir(relative_dir: str, required: bool = True):
            setup_py = ROOT / relative_dir / 'setup.py'
            if not setup_py.exists():
                self.announce(f'[SKIP] {relative_dir}/setup.py not found', level=2)
                return

            self.announce(f'[BUILD] {relative_dir}', level=2)
            try:
                subprocess.check_call([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=str(setup_py.parent))
            except subprocess.CalledProcessError:
                if required:
                    self.announce(f'[ERROR] Failed building required op: {relative_dir}', level=4)
                    raise
                self.announce(f'[WARN] Failed building optional op: {relative_dir}', level=2)

        build_plan = [
            ('rcdfnet/ops/iou3d', True),
            ('rcdfnet/ops/voxel', True),
            ('rcdfnet/ops/spconv', True),
            ('rcdfnet/ops/bev_pool_v2', True),
        ]

        for relative_dir, required in build_plan:
            build_ext_in_dir(relative_dir, required)

        ingroup_setup = ROOT / 'setup_ingroup_indices.py'
        if not ingroup_setup.exists():
            raise FileNotFoundError(f'Missing ingroup setup script: {ingroup_setup}')

        self.announce('[BUILD] ingroup_indices', level=2)
        subprocess.check_call(
            [sys.executable, 'setup_ingroup_indices.py', 'build_ext', '--inplace'],
            cwd=str(ROOT),
        )
        self.announce('[OK] build_ops finished', level=2)


class BuildExtWithRcdbfOps(build_ext):
    """Hook rcdfnet op builds into the standard build_ext command."""

    def run(self):
        run_build_ops_once(self)
        super().run()


class BuildPyWithRcdbfOps(build_py):
    """Ensure editable installs also trigger op compilation."""

    def run(self):
        run_build_ops_once(self)
        super().run()


setup(
    name='rcdfnet-main',
    version='0.1.0',
    description='Standalone RCDFNet package migrated from mmdet3d branch',
    packages=find_packages(include=['rcdfnet', 'rcdfnet.*', 'plot', 'plot.*']),
    include_package_data=True,
    cmdclass={
        'build_rcdfnet_ops': BuildRcdbfOps,
        'build_py': BuildPyWithRcdbfOps,
        'build_ext': BuildExtWithRcdbfOps,
    },
)
