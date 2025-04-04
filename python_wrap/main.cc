#include "bundle_adjuster.h"
#include "nanobind/nanobind.h"

NB_MODULE(bundlefit_ext, module) {
    bundlefit::WrapBundleAdjuster(module);
}
