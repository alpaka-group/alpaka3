#pragma once

#include "idx_map_data.hpp"

#include <alpaka/alpaka.hpp>

#include <sciplot/sciplot.hpp>

#include <string>

void drawGraphPNG(alpaka::concepts::IDataSource<AccessData> auto data, std::string const filename)
{
    static_assert(data.dim() == 1);

    sciplot::Plot2D plot;
    plot.xlabel("data ID");
    plot.ylabel("processing IDs");
    plot.xrange(0.0, static_cast<double>(data.getExtents().product()));
    plot.legend().atOutsideTopRight();

    sciplot::Vec const x = sciplot::range(0, data.getExtents().product());

    sciplot::Vec frame_elem(data.getExtents().product());
    sciplot::Vec thread_id(data.getExtents().product());
    sciplot::Vec block_id(data.getExtents().product());

    for(auto i = 0; i < data.getExtents()[0]; ++i)
    {
        thread_id[i] = data[i].thread_id.product();
        block_id[i] = data[i].block_id.product();
    }

    auto frame_elem_offset = thread_id.max();

    for(auto i = 0; i < data.getExtents()[0]; ++i)
    {
        frame_elem[i] = data[i].frame_elem.product() + frame_elem_offset + 10.0;
    }

    plot.drawCurve(x, frame_elem).label("Frame Element");
    plot.drawBoxes(x, thread_id).label("Thread Index");
    plot.drawBoxes(x, block_id).label("Block Index");

    sciplot::Figure fig = {{plot}};
    sciplot::Canvas canvas = {{fig}};

    constexpr auto scale = 100;
    canvas.size(scale * 81, scale * 50);

    canvas.save(filename + ".svg");
}
