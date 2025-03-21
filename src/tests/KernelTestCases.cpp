/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed,
 * transmitted, transcribed, stored in a retrieval system, or translated into any human
 * or computer language in any form by any means,electronic, mechanical, manual or otherwise,
 * or disclosed to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose
 * only)
 **************************************************************************************************/

#include "tests/KernelTestCases.h"

#include "gtest/gtest.h"
#include "tests/PIMKernel.h"

/*
 * PIMKernelTest:
 * functionality test for PIM Kernel function
 */

using namespace DRAMSim;

TEST_F(PIMKernelFixture, gemv)
{
    shared_ptr<PIMKernel> kernel = make_pim_kernel();

    uint32_t batch_size = 1;
    // CURT'S NOTE: Modify this to change the data used for the GEMV Kernel Test (functional verification).
    // This will lead to the application pulling the corresponding npy files in the data/gemv foler.
    // Then run ./sim --gtest_filter=PIMKernelFixture.gemv
    uint32_t output_dim = 64;
    uint32_t input_dim = 256;

    DataDim *dim_data = new DataDim(KernelType::GEMV, batch_size, output_dim, input_dim, true);
    dim_data->printDim(KernelType::GEMV);

    reduced_result_ = new BurstType[dim_data->dimTobShape(output_dim)];
    result_ = getResultPIM(KernelType::GEMV, dim_data, kernel, result_);

    testStatsClear();
    expectAccuracy(KernelType::GEMV, output_dim, dim_data->output_npbst_,
                   dim_data->getNumElementsPerBlocks());

    delete[] result_;
    delete[] reduced_result_;
    delete dim_data;
}
