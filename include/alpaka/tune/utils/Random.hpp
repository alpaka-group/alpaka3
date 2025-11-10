//
// Created by tim on 03.07.25.
//
#ifndef RNG_H
#define RNG_H
#include <random>

namespace alpaka::tune::strategy
{


    class RNG
    {
    public:
        static std::mt19937& get()
        {
            static RNG instance;
            return instance.rng_;
        }

    private:
        RNG()
        {
            std::random_device rd;
            rng_ = std::mt19937(rd());
        }

        std::mt19937 rng_;
    };
} // namespace alpaka::tune::strategy
#endif // RNG_H
