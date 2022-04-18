
#include <iostream>
#include <cstdlib>
#include <vector>
#include <fstream>

#include "../lib/trend.hpp"

int main(int argc, char *argv[])
{
    TrendHFT spxl_trendHFT("SPXL");
    spxl_trendHFT.build();

    return 0;
}

