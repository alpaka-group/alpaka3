/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: Apache-2.0
 */

#include <alpaka/alpaka.hpp>

#include <boost/program_options.hpp>
#include <iostream>
#include <vector>

namespace alpaka::example::radixSort
{
}

auto main(int argc, char* argv[]) -> int
{
    using namespace alpaka;
    using namespace alpaka::example::radixSort;
    namespace po = boost::program_options;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("input", po::value<std::vector<int>>()->multitoken(), "Input data for the sorting algorithm")
    ;

    po::variables_map vm;

    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (po::error const& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    if(vm.count("help")) {
        std::cout << desc << "\n";

        return 0;
    }

    if (!vm.count("input")) {
        std::cerr << "No input data provided!" << std::endl;
        return 1;
    }

    // TODO
    for (auto i: vm["input"].as<std::vector<int>>())
            std::cout << i << '\n';

    return 0;
}
