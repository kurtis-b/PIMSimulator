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

#include "tests/PIMKernel.h"

#include <iomanip>
#include <string>

#include "AddressMapping.h"
#include "tests/PIMCmdGen.h"

void PIMKernel::runPIM()
{
    while (mem_->hasPendingTransactions())
    {
        cycle_++;
        mem_->update();
    }
}

uint64_t PIMKernel::getCycle()
{
    return cycle_;
}

void PIMKernel::parkIn()
{
    addBarrier();
    for (int &ch_idx : pim_chans_)
    {
        for (int &ra_idx : pim_ranks_)
        {
            for (int bank_idx = 0; bank_idx < num_banks_; bank_idx++)
            {
                for (int bg_idx = 0; bg_idx < num_bank_groups_; bg_idx++)
                {
                    if (bank_idx != bg_idx)
                        continue; // Skip when indexes don't match because the expectation is that there's one bank per bank-group
                    string str = "PARK_IN_";
                    if (bg_idx == 0 && bank_idx == 0)
                        str = "START_" + str;
                    else if (bg_idx == 15 && bank_idx == 15)
                        str = "END_" + str;
                    mem_->addTransaction(
                        false,
                        pim_addr_mgr_->addrGen(ch_idx, ra_idx, bg_idx, bank_idx, (1 << 13), 0), str,
                        &null_bst_);
                }
            }
        }
    }
    addBarrier();
}

void PIMKernel::parkOut()
{
    for (int &ch_idx : pim_chans_)
    {
        for (int &ra_idx : pim_ranks_)
        {
            for (int bank_idx = 0; bank_idx < num_banks_; bank_idx++)
            {
                for (int bg_idx = 0; bg_idx < num_bank_groups_; bg_idx++)
                {
                    if (bank_idx != bg_idx)
                        continue; // Skip when indexes don't match because the expectation is that there's one bank per bank-group
                    string str = "PARK_OUT_";
                    if (bg_idx == 0 && bank_idx == 0)
                        str = "START_" + str;
                    else if (bg_idx == 15 && bank_idx == 15)
                        str = "END_" + str;
                    mem_->addTransaction(
                        false,
                        pim_addr_mgr_->addrGen(ch_idx, ra_idx, bg_idx, bank_idx, (1 << 13), 0), str,
                        &null_bst_);
                }
            }
        }
    }
    addBarrier();
}

void PIMKernel::addTransactionAll(bool is_write, int bg_idx, int bank_idx, int row, int col,
                                  const string tag, BurstType *bst, bool use_barrier, int num_loop)
{
    for (int &ch_idx : pim_chans_)
    {
        for (int &ra_idx : pim_ranks_)
        {
            unsigned local_row = row;
            unsigned local_col = col;
            for (int i = 0; i < num_loop; i++)
            {
                uint64_t addr = pim_addr_mgr_->addrGenSafe(ch_idx, ra_idx, bg_idx, bank_idx,
                                                           local_row, local_col);
                // std::cout << "Add transaction to all with addr: " << std::hex << addr << std::dec << ", tag: " << tag << std::endl;
                bool success = (tag != "") ? mem_->addTransaction(is_write, addr, tag, bst)
                                           : mem_->addTransaction(is_write, addr, bst);
                if (!success)
                {
                    DEBUG("Failed to add transaction to multi-ch mem system: " << addr);
                }
                else if (row == pim_abmr_ra_)
                {
                    DEBUG("Successfully added transaction for ab sequence row");
                }
                local_col++;
            }
        }
    }

    if (use_barrier)
        addBarrier();
}

void PIMKernel::addTransactionAll(bool is_write, int bg_idx, int bank_idx, int row, int col,
                                  BurstType *bst, bool use_barrier, int num_loop)
{
    addTransactionAll(is_write, bg_idx, bank_idx, row, col, "", bst, use_barrier, num_loop);
}

void PIMKernel::addBarrier()
{
    for (int &ch_idx : pim_chans_)
        mem_->addBarrier(ch_idx);
}

void PIMKernel::changePIMMode(dramMode curMode, dramMode nextMode)
{
    if (curMode == dramMode::SB && nextMode == dramMode::HAB)
    {
        DEBUG("SETTING PACKETS TO SWITCH TO HAB");
        DEBUG("Adding transaction to all banks/bgs with index 0, 1, 8, 9, for row " << std::hex << pim_abmr_ra_ << std::dec << ", col: " << std::hex << pim_abmr_ca_ << std::dec);
        addTransactionAll(true, 0, 0, pim_abmr_ra_, pim_abmr_ca_, "START_SB_TO_HAB_", &null_bst_);
        addTransactionAll(true, 1, 1, pim_abmr_ra_, pim_abmr_ca_, &null_bst_);
        DEBUG("num banks: " << num_banks_);
        if (num_banks_ >= 2)
        {
            addTransactionAll(true, 8, 8, pim_abmr_ra_, pim_abmr_ca_, &null_bst_);
            addTransactionAll(true, 9, 9, pim_abmr_ra_, pim_abmr_ca_, "END_SB_TO_HAB_", &null_bst_);
        }
    }
    else if (curMode == dramMode::HAB)
    {
        if (nextMode == dramMode::SB)
        {
            addTransactionAll(true, 0, 0, pim_sbmr_ra_, pim_abmr_ca_, "START_HAB_TO_SB", &null_bst_);
            addTransactionAll(true, 1, 1, pim_sbmr_ra_, pim_abmr_ca_, "END_HAB_TO_SB", &null_bst_);
        }
        else if (nextMode == dramMode::HAB_PIM)
        {
            addTransactionAll(true, 0, 0, pim_reg_ra_1, 0x0, "PIM", &bst_hab_pim_);
        }
    }
    else if (curMode == dramMode::HAB_PIM && nextMode == dramMode::HAB)
        addTransactionAll(true, 0, 0, pim_reg_ra_1, 0x0, "PIM", &bst_hab_);

    addBarrier();
}

/*
// FIXME : FIX size of srf_bst_. if ch_model is bigger than memory channel, it is not defined.
void PIMKernel::preprocessSrf(NumpyBurstType* input_npbst, fp16** params, int burst_offset,
                              int num_srf_usage)
{
    int ch_idx = 0;
    int ra_idx = 0;
    int burst_idx = 0;
    int num_stride_reg = 2;
    srf_bst_ = new BurstType[num_pim_chans_ * num_pim_ranks_];

    for (int ch_model = 0; ch_model < input_npbst->bShape[0]; ch_model++)
    {
        srf_bst_[ch_idx * num_pim_ranks_ + ra_idx].fp16Data_[burst_idx] =
            params[ch_model][0]; // scale
        srf_bst_[ch_idx * num_pim_ranks_ + ra_idx].fp16Data_[burst_idx + 1] =
            params[ch_model][1]; // gamma
        srf_bst_[ch_idx * num_pim_ranks_ + ra_idx].fp16Data_[burst_idx + 8] =
            params[ch_model][2]; // shift
        srf_bst_[ch_idx * num_pim_ranks_ + ra_idx].fp16Data_[burst_idx + 9] =
            params[ch_model][3]; // beta

        ra_idx++;
        if (ra_idx >= num_pim_ranks_)
        {
            ra_idx = 0;
            ch_idx++;
        }
        if (ch_idx >= num_pim_chans_)
        {
            ch_idx = 0;
            burst_idx += num_stride_reg;
        }
        if (burst_idx >= 8)
        {
            cout << "error: this is not defined" <<endl;
        }
    }
}

void PIMKernel::programSrf()
{
    for (int ch_idx = 0; ch_idx < num_pim_chans_; ch_idx++)
    {
        for (int ra_idx = 0; ra_idx < num_pim_ranks_; ra_idx++)
        {
            mem_->addTransaction(true,
                                 pim_addr_mgr_->addrGen(ch_idx, ra_idx, 0, 0, pim_reg_ra_, 0x1),
                                 &srf_bst_[ch_idx * num_pim_ranks_ + ra_idx]);
        }
    }
    addBarrier();
}
*/

void PIMKernel::programCrf(vector<PIMCmd> &cmds)
{
    PIMCmd nop_cmd(PIMCmdType::NOP, 0);
    // std::cout << "cmds.size: " << cmds.size() << std::endl;
    for (int i = 0; i < 16; i++)
    {
        if (i * 8 >= cmds.size())
        {
            break;
        }
        crf_bst_[i].set(nop_cmd.toInt(), nop_cmd.toInt(), nop_cmd.toInt(), nop_cmd.toInt(),
                        nop_cmd.toInt(), nop_cmd.toInt(), nop_cmd.toInt(), nop_cmd.toInt());
        for (int j = 0; j < 8; j++)
        {
            if (i * 8 + j >= cmds.size())
            {
                break;
            }
            crf_bst_[i].u32Data_[j] = cmds[i * 8 + j].toInt();
            // std::cout << "cmds[" << i << " * 8 + " << j << "].toStr()" << cmds[i * 8 + j].toStr()  << std::endl;
        }
        addTransactionAll(true, 0, 0, pim_reg_ra_1, 0x4 + i, "PROGRAM_CRF", &(crf_bst_[i]));
    }
    addBarrier();
}

void PIMKernel::setControl(BurstType *bst, bool pim_op, int crf_toggle_cond, bool grfA_zero,
                           bool grfB_zero)
{
    bst->u8Data_[0] = pim_op;
    bst->u8Data_[16] = crf_toggle_cond;
    bst->u8Data_[20] = grfA_zero;
    bst->u8Data_[21] = grfB_zero;
}

unsigned PIMKernel::getResultColGemv(int input_dim, int output_dim)
{
    int num_output_tiles = ceil(((double)output_dim / (num_banks_)) / num_pim_chans_);
    int num_input_tiles = input_dim;

    return num_output_tiles * num_input_tiles;
}

void PIMKernel::changeBank(pimBankType pb_type, int &ch_idx, int &ra_idx, int &bg_idx,
                           int &bank_idx, unsigned &starting_row, unsigned &starting_col,
                           unsigned &row, unsigned &col)
{
    bank_idx += (pb_type == pimBankType::ALL_BANK) ? 1 : (num_banks_ / num_pim_blocks_);
    bg_idx += 1; // Each bank will be its own bank group

    if (bank_idx >= num_banks_ && bg_idx >= num_bank_groups_)
    {
        bank_idx = 0;
        bg_idx = 0;
        if (++ra_idx >= num_pim_ranks_)
        {
            ra_idx = 0;
            if (++ch_idx >= num_pim_chans_)
            {
                ch_idx = 0;
                starting_row = row;
                starting_col = col;
            }
        }
    }
}

void PIMKernel::preloadGemv(NumpyBurstType *operand, unsigned starting_row, unsigned starting_col)
{
    // Input tiles will traverse over the columns of the wt mtx and correspond to the tiles of the input activation vector
    int num_input_tiles = operand->bShape[1]; // = (# wt mtx cols * total bits per element) / (device width * burst length)
    // Output tiles will traverse over the rows of the wt mtx and the correspond to the tiles of the output activation vector
    int num_output_tiles = ceil(((double)operand->bShape[0] / (num_banks_)) / num_pim_chans_); // Tile the output vector calculation across the banks across the channels, so this is for 1 bank/1 channel

    int ch_idx = 0, ra_idx = 0, bg_idx = 0, bank_idx = 0;
    unsigned row = 0, col = 0;
    uint64_t addr;

    // Assuming that a weight matrix column fits within one bank column
    // Will need to look into cases where the weight matrix column is larger than the bank column
    for (int y = 0; y < operand->bShape[0]; y += num_output_tiles)
    {
        for (int tiled_y = 0; tiled_y < num_output_tiles; tiled_y++)
        {
            for (int x = 0; x < num_input_tiles; x++) // Each x will contain a tile of 16 fp16 that can be referenced from the operand
            {
                addr = pim_addr_mgr_->addrGenSafe(ch_idx, ra_idx, bg_idx, bank_idx,
                                                  row, col);

                // std::cout << "addr: " << std::hex << addr << std::dec << " from ch_idx: " << ch_idx << ", ra_idx: " << ra_idx;
                // std::cout << ", bg_idx: " << bg_idx << ", bank_idx: " << bank_idx << ", row: " << row;
                // std::cout << ", col: " << col << std::endl;

                int d_idx = (y + tiled_y) * operand->bShape[1] + x;
                mem_->addTransaction(true, addr, &operand->bData[d_idx]);

                // std::cout << "d_idx: " <<  d_idx << " from (y + tiled_y) * operand->bShape[1] + x, which was";
                // std::cout << "(" << y << " + " << tiled_y << ") * " << operand->bShape[1] << " + " << x << std::endl;
                col++;
            }
        }
        // Each bank will be its own bank group
        bank_idx += 1;
        bg_idx += 1;
        row = 0; // reset row for a new bank
        col = 0; // reset col for a new bank

        if (bank_idx >= num_banks_ && bg_idx >= num_bank_groups_)
        {
            bank_idx = 0;
            bg_idx = 0;
            if (++ra_idx >= num_pim_ranks_)
            {
                ra_idx = 0;
                row = 0; // reset row for a new channel
                col = 0; // reset col for a new channel
                if (++ch_idx >= num_pim_chans_)
                {
                    ch_idx = 0;
                }
            }
        }
    }
}

void PIMKernel::preloadNoReplacement(NumpyBurstType *operand, unsigned starting_row,
                                     unsigned starting_col)
{
    uint64_t init_addr = pim_addr_mgr_->addrGenSafe(0, 0, 0, 0, starting_row, starting_col);

    for (int x = 0; x < operand->getTotalDim(); x++)
    {
        uint64_t addr = init_addr + x * transaction_size_;
        mem_->addTransaction(true, addr, &operand->bData[x]);
    }
}
/*
void PIMKernel::preloadEltwise(NumpyBurstType* operand, pimBankType pb_type,
                              unsigned starting_row, unsigned starting_col)
{
   int ch_idx = 0;
   int ra_idx = 0;
   int bg_idx = 0;
   int bank_idx = 0;
   int bank_offset =  (int)pb_type % 2;
   uint64_t addr_op;
   int dim_operand = operand->getTotalDim();

   for (int x=0; x < dim_operand; x+=num_grf_)
   {
       unsigned col = starting_col;
       unsigned row = starting_row;

       for (int grf_idx = 0; grf_idx < num_grf_; grf_idx++)
       {
           addr_op = pim_addr_mgr_->addrGenSafe(ch_idx, ra_idx, bg_idx, bank_idx + bank_offset, row,
                                                col);
           mem_->addTransaction(true, addr_op, &operand->bData[x + grf_idx]);
           col++;
       }
       changeBank(pb_type, ch_idx, ra_idx, bg_idx, bank_idx, starting_row, starting_col, row, col);
   }
}
*/
void PIMKernel::executeGemv(NumpyBurstType *w_data, NumpyBurstType *i_data, bool is_tree)
{
    // CURT'S NOTE: GEMV tree is not suppported, so not changing anything for parts related to it
    if (is_tree)
        cerr << "Not implemented!" << std::endl;

    // second dimension of numpy burst weight mtx shape will contain the number of MAC commands (16 fp16 macs at a time) needed for 1 output element
    // 1 bank col = (# wt mtx cols * total bits per element) / (device width * burst length)
    // Num pim exectuions is the number times to run PIM to use the full global buffer, basically how many output elems calculated using the full global buffer
    int num_repeat_kernel_insts = num_grfA_ / w_data->bShape[1];
    // Handle the case where the global buffer is larger than the weight matrix column size
    // In this case, the input vector is copied to the global buffer more than once to fill it
    if (num_repeat_kernel_insts == 0)
        num_repeat_kernel_insts = 1;
    int num_full_pim_executions = w_data->bShape[1] / num_grfA_;
    if (num_full_pim_executions == 0)
        num_full_pim_executions = 1;
    int num_grfa_to_use = num_grfA_ / num_repeat_kernel_insts;
    vector<PIMCmd> pim_cmds = PIMCmdGen::getPIMCmds(KernelType::GEMV, num_grfa_to_use, num_repeat_kernel_insts, num_full_pim_executions);
    setControl(&bst_hab_pim_, true, getToggleCond(), false, true);
    parkIn();
    changePIMMode(dramMode::SB, dramMode::HAB);
    programCrf(pim_cmds);

    for (auto &pim_cmd : pim_cmds)
        DEBUG("pim_cmd: " << pim_cmd.toStr());

    // Tile the output vector calculation across the banks across the channels, so loop below is for 1 bank/1 channel
    int num_output_tiles = ceil(((double)w_data->bShape[0] / (num_banks_)) / num_pim_chans_);
    int num_batch = i_data->bShape[0];
    int num_y_tiles = num_output_tiles / num_repeat_kernel_insts;
    if (num_y_tiles == 0)
        num_y_tiles = 1;
    for (int b = 0; b < num_batch; b++)
    {
        for (int tiled_y = 0; tiled_y < num_y_tiles; tiled_y++)
        {
            changePIMMode(dramMode::HAB, dramMode::HAB_PIM); // PC reset.
            for (int offset = 0; offset < num_full_pim_executions; offset++)
            {
                if (tiled_y == 0 || num_full_pim_executions > 1)
                {
                    // Input upload to GRF. The below loop should fill the global buffer (all GRF A's) even if
                    // the input vector is smaller than the size of the global buffer (< 1024 fp16 elements)
                    for (int ch_idx = 0; ch_idx < num_pim_chans_; ch_idx++)
                    {
                        for (int ra_idx = 0; ra_idx < num_pim_ranks_; ra_idx++)
                        {
                            for (int repeat = 0; repeat < num_repeat_kernel_insts; repeat++)
                            {
                                for (int inp_tile_idx = 0; inp_tile_idx < num_grfA_ / num_repeat_kernel_insts; inp_tile_idx++)
                                {
                                    string str = "WRIO_TO_GRFA_";
                                    // The input vector will be broadcasted to GRF A's of each pim block. This is to mimic the global buffer in SK Hynix
                                    int g_idx = (repeat * w_data->bShape[1]) % num_grfA_ + inp_tile_idx;
                                    uint64_t addr = pim_addr_mgr_->addrGen(ch_idx, ra_idx, 0, 0, pim_reg_ra_2, g_idx);

                                    int input_idx = (offset * num_grfA_) % w_data->bShape[1] + inp_tile_idx;
                                    mem_->addTransaction(true, addr, str, &i_data->bData[input_idx]);
                                    // std::cout << "Add transaction to mem sys with addr: " << std::hex << addr << std::dec << ", input_idx: " << input_idx << std::endl;
                                }
                            }
                            mem_->addBarrier(ch_idx);
                        }
                    }
                }

                // Execute MACs
                for (int tiled_mac_iter = 0; tiled_mac_iter < num_grfA_; tiled_mac_iter++) // Each x will contain a tile of 16 fp16 that can be referenced from the operand
                {
                    int row = tiled_y * num_full_pim_executions + offset;
                    int col = tiled_mac_iter;
                    // This runs MAC for all banks (pim blocks) with the GRF A (input vector tile) broadcasted to the banks (based on the commands in CRF),
                    // and the corresponding row/col executed for MAC. Each pim block will contain one element in its GRF B as the output element.
                    addTransactionAll(false, 0, 0, row, col, "MAC_", &null_bst_, true);
                    // std::cout << "Add transaction for MAC with row " << row << " col " << col << std::endl;
                    if ((offset * num_grfA_ + (tiled_mac_iter + 1)) % w_data->bShape[1] == 0)
                    {
                        // Column should start at 0, so not adding 1 to tiled_mac_iter when dividing
                        // std::cout << "Writing GRFB to col " << tiled_y * num_repeat_kernel_insts + tiled_mac_iter / w_data->bShape[1] << std::endl;
                        addTransactionAll(true, 0, 0, pim_reg_ra_1 >> 1, tiled_y * num_repeat_kernel_insts + tiled_mac_iter / w_data->bShape[1], "GRFB_TO_BANK_", &null_bst_, true);
                    }
                }
            }
            changePIMMode(dramMode::HAB_PIM, dramMode::HAB); // for grfBReset
        }
    }
    changePIMMode(dramMode::HAB, dramMode::SB);
    parkOut();
}

void PIMKernel::readResult(BurstType *resultBst, pimBankType pb_type, int output_dim,
                           uint64_t base_addr, unsigned starting_row, unsigned starting_col)
{
    int num_output_tiles = ceil(((double)output_dim / (num_banks_)) / num_pim_chans_); // Tile the output vector calculation across the banks across the channels, so this is for 1 bank/1 channel

    int ch_idx = 0, ra_idx = 0, bg_idx = 0, bank_idx = 0;
    unsigned row = base_addr, col = 0;
    uint64_t addr;

    for (int y = 0; y < output_dim; y += num_output_tiles)
    {
        for (int tiled_y = 0; tiled_y < num_output_tiles; tiled_y++)
        {
            addr = pim_addr_mgr_->addrGenSafe(ch_idx, ra_idx, bg_idx, bank_idx, row,
                                              col);

            // std::cout << "addr: " << std::hex << addr << std::dec << " from ch_idx: " << ch_idx << ", ra_idx: " << ra_idx;
            // std::cout << ", bg_idx: " << bg_idx << ", bank_idx: " << bank_idx << ", row: " << row;
            // std::cout << ", col: " << col << std::endl;

            mem_->addTransaction(false, addr, "output", &resultBst[y + tiled_y]);

            // std::cout << "d_idx: " <<  y + tiled_y << " from y + tiled_y, which was ";
            // std::cout << y << " + " << tiled_y << std::endl;
            col++;
        }
        // Each bank will be its own bank group
        bank_idx += 1;
        bg_idx += 1;
        row = base_addr; // reset row for a new bank
        col = 0;         // reset col for a new bank

        if (bank_idx >= num_banks_ && bg_idx >= num_bank_groups_)
        {
            bank_idx = 0;
            bg_idx = 0;
            if (++ra_idx >= num_pim_ranks_)
            {
                ra_idx = 0;
                row = base_addr; // reset row for a new channel
                col = 0;         // reset col for a new channel
                if (++ch_idx >= num_pim_chans_)
                {
                    ch_idx = 0;
                }
            }
        }
    }
}

void PIMKernel::executeEltwise(int dim, pimBankType pb_type, KernelType ktype, int input0_row,
                               int result_row, int input1_row)
{
    cerr << "Not implemented!" << std::endl;
}

void PIMKernel::computeAddOrMul(int num_tile, int input0_row, int result_row, int input1_row)
{
    cerr << "Not implemented!" << std::endl;
}

void PIMKernel::computeRelu(int num_tile, int input0_row, int result_row)
{
    cerr << "Not implemented!" << std::endl;
}

void PIMKernel::readData(BurstType *bst_data, size_t bst_cnt, unsigned starting_row,
                         unsigned starting_col)
{
    uint64_t init_addr = pim_addr_mgr_->addrGenSafe(0, 0, 0, 0, starting_row, starting_col);

    for (uint64_t addr = init_addr, i = 0; i < bst_cnt; addr += transaction_size_, i++)
    {
        mem_->addTransaction(false, addr, &bst_data[i]);
    }
}

void PIMKernel::adderTree(BurstType *result, int output_dim, int num_tile, int step, fp16 *temp)
{
    if (num_tile == 1)
        return;

    int iter = num_tile / 2;
    if (step == 0)
    {
        for (int i = 0; i < iter; i++)
        {
            temp[i] = result[2 * i * output_dim].fp16AdderTree() +
                      result[(2 * i + 1) * output_dim].fp16AdderTree();
        }
    }
    else
    {
        for (int i = 0; i < iter; i++)
            temp[i] = temp[i * 2] + temp[i * 2 + 1];

        if (num_tile % 2 == 1)
            temp[iter] = temp[num_tile];
    }

    adderTree(result, output_dim, ceil(double(num_tile) / (double)2), step + 1, temp);

    return;
}
