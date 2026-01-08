from conan import ConanFile
from conan.tools.cmake import cmake_layout
from conan.errors import ConanInvalidConfiguration


class ExampleRecipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("opencv/4.12.0")
        self.requires("onnxruntime/1.23.2")
        # Force cpuinfo version to satisfy ONNX Runtime
        self.requires("cpuinfo/cci.20250110", override=True)

    def layout(self):
        cmake_layout(self)

    def validate(self):
        # Check if the compiler version in the profile supports C++20
        if self.settings.compiler.get_safe("cppstd"):
            if int(self.settings.compiler.cppstd.value) < 20:
                raise ConanInvalidConfiguration("This project requires C++20")

    def configure(self):
        # Force the requirement for C++20 in the package settings
        self.settings.compiler.cppstd = "20"
        self.options["opencv"].ximgproc = True
        self.options["opencv"].with_ximgproc = True
        self.options["opencv"].with_protobuf = False  # for protobuf conflict
        self.options["opencv"].with_eigen = False     # disable Eigen
        