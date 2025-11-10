//
// Created by tim on 03.07.25.
//
#ifndef RNG_H
#define RNG_H
#include <random>

namespace alpaka::onHost::tune::strategy::helper
{


    class StdMTRandomDevice
    {
    public:
        static std::mt19937& get()
        {
            static StdMTRandomDevice instance;
            return instance.rng_;
        }

    private:
        StdMTRandomDevice()
        {
            std::random_device rd;
            rng_ = std::mt19937(rd());
        }

        std::mt19937 rng_;
    };
} // namespace alpaka::onHost::tune::strategy::helper
#endif // RNG_H
