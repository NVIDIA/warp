// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace cuBQL {
  namespace gpuBuilder_impl {

    //#define CUBQL_PROFILE 1

#if CUBQL_PROFILE
    struct Profile {
      void setName(std::string name, int sub=-1)
      {
        if (sub >= 0) {
          char suff[1000];
          sprintf(suff,"[%2i]",sub);
          this->name = name+suff;
        } else
          this->name = name;
      }
      ~Profile() { ping(); }
      
      void start() {
        t0 = getCurrentTime();
      }
      void sync_start() {
        CUBQL_CUDA_SYNC_CHECK();
        start();
      }
      void sync_stop() {
        CUBQL_CUDA_SYNC_CHECK();
        stop();
      }
      void stop(bool do_ping = false) {
        double t1 = getCurrentTime();
        t_sum += (t1-t0);
        count ++;
        if (do_ping) ping();
      }
      void ping()
      {
        if (count)
          std::cout << "#PROF " << name << " = " << prettyDouble(t_sum / count) << std::endl;
      }
      double t0 = 0.;
      double t_sum = 0.;
      int count = 0;
      std::string name = "";
    };
#endif
    
  }
}
