/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */
 
#include <alpaka/alpaka.hpp>
#include <iostream>

int main(int argc, char **argv) {
  auto computeDevSpec =
      alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
  auto deviceSelector = alpaka::onHost::makeDeviceSelector(computeDevSpec);

  auto num_devices = deviceSelector.getDeviceCount();
  std::cout << "Number of available Devices: " << num_devices << "\n";

  if (num_devices == 0) {
    return 1;
  }

  auto device = deviceSelector.makeDevice(0);
  std::cout << "Device 0: " << device.getName() << "\n";

  return 0;
}
