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
                DEBUG("addr: " << std::hex << addr << std::dec << ", tag: " << tag);
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
            addTransactionAll(true, 0, 0, pim_reg_ra_, 0x0, "PIM", &bst_hab_pim_);
        }
    }
    else if (curMode == dramMode::HAB_PIM && nextMode == dramMode::HAB)
        addTransactionAll(true, 0, 0, pim_reg_ra_, 0x0, "PIM", &bst_hab_);

    addBarrier();
}

/*
void PIMKernel::preprocessBn(NumpyBurstType* mean_npbst, NumpyBurstType* var_npbst,
                             NumpyBurstType* gamma_npbst, NumpyBurstType* beta_npbst,
                             NumpyBurstType* input_npbst, fp16** params, float eps)
{
    for (int i = 0; i < input_npbst->bShape[0]; i++)
    {
        params[i][0] = 1 / sqrt((float)var_npbst->getBurst(i / 16).fp16Data_[i % 16] + eps);
        params[i][1] = gamma_npbst->getBurst(i / 16).fp16Data_[i % 16];
        params[i][2] = -mean_npbst->getBurst(i / 16).fp16Data_[i % 16] /
                       sqrt((float)var_npbst->getBurst(i / 16).fp16Data_[i % 16] + eps);
        params[i][3] = beta_npbst->getBurst(i / 16).fp16Data_[i % 16];
    }
}

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
    DEBUG("cmds.size: " << cmds.size());
    for (int i = 0; i < 4; i++)
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
            DEBUG("cmds[" << i << " * 8 + " << j << "].toStr()" << cmds[i * 8 + j].toStr());
        }
        addTransactionAll(true, 1, 1, pim_reg_ra_, 0x4 + i, "PROGRAM_CRF", &(crf_bst_[i]));
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
    int num_output_tiles = ceil(((double)output_dim / (num_total_pim_blocks_)) / num_grfB_);
    int num_input_tiles = ceil((double)input_dim / (double)num_grfA_);

    return num_output_tiles * num_input_tiles / 2 * num_grfA_ * num_grfB_;
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
    int wt_tile_cols = operand->bShape[1];
    int wt_tile_rows = pim_addr_mgr_->num_cols_per_bl_ / operand->bShape[1];

    int ch_idx = 0, ra_idx = 0, bg_idx = 0, bank_idx = 0;
    unsigned row = 0, col = 0;
    uint64_t addr;

    // Assuming that a weight matrix column fits within one bank column
    // Will need to look into cases where the weight matrix column is larger than the bank column
    for (int y = 0; y < operand->bShape[0] / wt_tile_rows; y++)
    {
        for (int tiled_y = 0; tiled_y < wt_tile_rows; tiled_y++)
        {
            for (int tiled_x = 0; tiled_x < wt_tile_cols; tiled_x++)
            {
                addr = pim_addr_mgr_->addrGenSafe(ch_idx, ra_idx, bg_idx, bank_idx,
                                                  row, col);

                DEBUG("addr: " << std::hex << addr << std::dec << " from ch_idx: " << ch_idx << ", ra_idx: " << ra_idx);
                DEBUG(", bg_idx: " << bg_idx << ", bank_idx: " << bank_idx << ", row: " << row);
                DEBUG(", col: " << col);

                int d_idx = (y * wt_tile_rows + tiled_y) * operand->bShape[1] + tiled_x;
                mem_->addTransaction(true, addr, &operand->bData[d_idx]);

                DEBUG("d_idx: " << d_idx << " from (y * wt_tile_rows + tiled_y) * operand->bShape[1] + tiled_x, which was");
                DEBUG("(" << y << " * " << wt_tile_rows << " + " << tiled_y << ") * " << operand->bShape[1] << " +  " << tiled_x);
                col++;
            }
        }
        // Each bank will be its own bank group
        bank_idx += 1;
        bg_idx += 1;

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
    int num_output_tiles = ceil(((double)w_data->bShape[0] / (num_total_pim_blocks_)) / num_grfB_);
    int num_batch = i_data->bShape[0];
    int zero_row = 1000;
    int wt_tile_cols = w_data->bShape[1];
    int wt_tile_rows = pim_addr_mgr_->num_cols_per_bl_ / w_data->bShape[1];

    // CURT'S NOTE: GEMV tree is not suppported, so not changing anything for parts related to it
    if (is_tree)
        cerr << "GEMV tree mode not supported!" << endl;

    vector<PIMCmd> pim_cmds =
        PIMCmdGen::getPIMCmds(KernelType::GEMV, num_grfA_, num_output_tiles, w_data->bShape[1]);
    setControl(&bst_hab_pim_, true, getToggleCond(), false, true);
    parkIn();
    changePIMMode(dramMode::SB, dramMode::HAB);
    programCrf(pim_cmds);

    DEBUG("num_output_tiles: " << num_output_tiles << ", num_batch: " << num_batch << ", wt_tile_cols: " << wt_tile_cols << ", wt_tile_rows: " << wt_tile_rows);
    for (auto &pim_cmd : pim_cmds)
    {
        DEBUG("pim_cmd: " << pim_cmd.toStr());
    }

    for (int y = 0; y < w_data->bShape[0] / wt_tile_rows / num_banks_; y++) // Split the workload across the banks
    {
        for (int tiled_y = 0; tiled_y < wt_tile_rows; tiled_y++)
        {
            for (int b = 0; b < num_batch; b++)
            {
                changePIMMode(dramMode::HAB, dramMode::HAB_PIM); // PC reset.

                for (int bank_idx = 0; bank_idx < 2; bank_idx++) // Only doing two banks since that's what Samsung does originally
                {
                    for (int ch_idx = 0; ch_idx < num_pim_chans_; ch_idx++)
                    {
                        for (int ra_idx = 0; ra_idx < num_pim_ranks_; ra_idx++)
                        {
                            // Input upload to GRF. It should be 1024 fp16 elements long, but it's 256 fp16 elements for now
                            // since I'm working on workload 64x256 for GEMV
                            for (int g_idx = 0; g_idx < num_grfA_; g_idx++)
                            {
                                string str = "WRIO_TO_GRF_";
                                uint64_t addr =
                                    pim_addr_mgr_->addrGen(ch_idx, ra_idx, bank_idx, bank_idx, pim_reg_ra_, 0x8 + g_idx);
                                int input_idx =
                                    b * w_data->bShape[1] + (tiled_y + 1) * wt_tile_rows + g_idx;

                                DEBUG("addr: " << addr << ", input_idx: " << input_idx);

                                mem_->addTransaction(true, addr, str, &i_data->bData[input_idx]);
                            }
                        }
                        mem_->addBarrier(ch_idx);
                    }

                    for (int tiled_x = 0; tiled_x < wt_tile_cols; tiled_x++)
                    {
                        addTransactionAll(false, bank_idx, bank_idx, y, tiled_y * wt_tile_cols + tiled_x, "MAC_", &null_bst_, true);
                    }
                    addTransactionAll(true, bank_idx, bank_idx, 1, 0, "GRFB_TO_BANK_", &null_bst_, true);
                }

                changePIMMode(dramMode::HAB_PIM, dramMode::HAB); // for grfBReset
            }
        }
    }
    changePIMMode(dramMode::HAB, dramMode::SB);
    parkOut();
}

void PIMKernel::computeGemv(NumpyBurstType *data, int num_input_tiles, int num_output_tiles,
                            int inputTile, int outputTile, int batchIdx, pimBankType pb_type)
{
    for (int ch_idx = 0; ch_idx < num_pim_chans_; ch_idx++)
    {
        for (int ra_idx = 0; ra_idx < num_pim_ranks_; ra_idx++)
        {
            // input upload to GRF
            for (int gidx = 0; gidx < num_grfA_; gidx++)
            {
                string str = "WRIO_TO_GRF_";
                uint64_t addr =
                    pim_addr_mgr_->addrGen(ch_idx, ra_idx, 0, 1, pim_reg_ra_, 0x8 + gidx);
                int input_idx =
                    batchIdx * num_grfA_ * num_input_tiles + inputTile * num_grfA_ + gidx;
                mem_->addTransaction(true, addr, str, &data->bData[input_idx]);
            }
            mem_->addBarrier(ch_idx);
        }
    }

    unsigned row = 0;
    unsigned col = (num_grfA_ * num_grfB_) * (inputTile / 2 + outputTile * num_input_tiles / 2);

    for (int c_idx = 0; c_idx < 64; c_idx += 8)
        addTransactionAll(false, 0, (int)pb_type, row, col + c_idx, "MAC_", &null_bst_, true,
                          num_grfA_);
}

void PIMKernel::readResult(BurstType *resultBst, pimBankType pb_type, int output_dim,
                           uint64_t base_addr, unsigned starting_row, unsigned starting_col)
{
    int ch_idx = 0;
    int ra_idx = 0;
    int bg_idx = 0;
    int bank_idx = 0;
    int bank_offset = (int)pb_type;
    uint64_t addr;

    for (int x = 0; x < output_dim; x += num_grf_)
    {
        unsigned row = starting_row;
        unsigned col = starting_col;

        for (int grf_idx = 0; grf_idx < num_grf_; grf_idx++)
        {
            addr = pim_addr_mgr_->addrGenSafe(ch_idx, ra_idx, bg_idx, bank_idx, row,
                                              col);
            mem_->addTransaction(false, base_addr + addr, "output", &resultBst[x + grf_idx]);
            col++;
        }
        changeBank(pb_type, ch_idx, ra_idx, bg_idx, bank_idx, starting_row, starting_col, row, col);
    }
}

void PIMKernel::executeEltwise(int dim, pimBankType pb_type, KernelType ktype, int input0_row,
                               int result_row, int input1_row)
{
    int num_tile = dim / (num_banks_ * num_pim_chans_ * num_pim_ranks_ * num_grf_);
    int num_jump_to_be_taken = num_tile - 1;
    vector<PIMCmd> pim_cmds = PIMCmdGen::getPIMCmds(ktype, num_jump_to_be_taken, 0, 0);

    setControl(&bst_hab_pim_, true, getToggleCond(pb_type), false, false);
    setControl(&bst_hab_, false, getToggleCond(pb_type), false, false);

    parkIn();
    changePIMMode(dramMode::SB, dramMode::HAB);
    programCrf(pim_cmds);
    changePIMMode(dramMode::HAB, dramMode::HAB_PIM);

    if (ktype == KernelType::ADD || ktype == KernelType::MUL)
        computeAddOrMul(num_tile, input0_row, result_row, input1_row);
    else if (ktype == KernelType::RELU)
        computeRelu(num_tile, input0_row, result_row);
    /*
       else if (ktype == KernelType::BN)
       computeBn(num_tile, input0_row, result_row);
     */

    changePIMMode(dramMode::HAB_PIM, dramMode::HAB);
    changePIMMode(dramMode::HAB, dramMode::SB);
    parkOut();
}

void PIMKernel::computeAddOrMul(int num_tile, int input0_row, int result_row, int input1_row)
{
    for (int i = 0; i < num_tile; i++)
    {
        int c = num_grf_ * i;
        for (int b = 0; b < 2; b++) // for even/odd banks, respectively
        {
            addTransactionAll(false, 0, b, input0_row, c, "BANK_TO_GRF_", &null_bst_, true,
                              num_grf_);
            addTransactionAll(false, 0, b, input1_row, c, "ADD", &null_bst_, true, num_grf_);
            addTransactionAll(true, 0, b, result_row, c, "GRF_TO_BANK", &null_bst_, true, num_grf_);
        }
    }
}

/*
void PIMKernel::computeBn(int num_tile, int input0_row, int result_row)
{
    for (int ch_idx = 0; ch_idx < num_pim_chans_; ch_idx++)
    {
        for (int ra_idx = 0; ra_idx < num_pim_ranks_; ra_idx++)
        {
            int srf_bst_num = (input0_row != result_row)? (ch_idx * num_pim_ranks_ + ra_idx) : 0;
            mem_->addTransaction(true, pim_addr_mgr_->addrGen(ch_idx, ra_idx, 0, 0, pim_reg_ra_,
                                       0x1), &srf_bst_[srf_bst_num]);
        }
    }
    addBarrier();

    if (input0_row != result_row)
        input0_row = result_row = 0;
    for (int i = 0; i < num_tile; i++)
    {
        for (int b = 0; b < 2; b++) // for even/ddd banks, respectively
        {
            addTransactionAll(false, 0, b, input0_row, num_grf_ * i, "MAD1", &null_bst_,
                              true, num_grf_);
            addTransactionAll(false, 0, b, input0_row, num_grf_ * i, "MAD2", &null_bst_,
                              true, num_grf_);
            addTransactionAll(true , 0, b, result_row, num_grf_ * i, "GRF_TO_BANK", &null_bst_,
                              true, num_grf_);
        }
    }
}
*/

void PIMKernel::computeRelu(int num_tile, int input0_row, int result_row)
{
    for (int i = 0; i < num_tile; i++)
    {
        int c = num_grf_ * i;
        addTransactionAll(false, 0, 0, input0_row, c, "FILL&ReLU", &null_bst_, true, num_grf_);
        addTransactionAll(true, 0, 0, result_row, c, "GRF_A_TO_EVEN_BANK", &null_bst_, true,
                          num_grf_);
        addTransactionAll(false, 0, 1, input0_row, c, "FILL&ReLU", &null_bst_, true, num_grf_);
        addTransactionAll(true, 0, 1, result_row, c, "GRF_B_TO_ODD_BANK", &null_bst_, true,
                          num_grf_);
    }
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
