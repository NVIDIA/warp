/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "builtin.h"

namespace wp
{

//----------------------------------------------------------------
// Generated adjoint for closest_point_edge_edge.
// See intersect.h for forward mode and comments.

static CUDA_CALLABLE void adj_closest_point_edge_edge(vec3 var_p1,
	vec3 var_q1,
	vec3 var_p2,
	vec3 var_q2,
	float32 var_epsilon,
	vec3 & adj_p1,
	vec3 & adj_q1,
	vec3 & adj_p2,
	vec3 & adj_q2,
	float32 & adj_epsilon,
	vec3 & adj_ret)
{
        //---------
    // primal vars
    vec3 var_0;
    vec3 var_1;
    vec3 var_2;
    float32 var_3;
    float32 var_4;
    float32 var_5;
    const float32 var_6 = 0.0;
    float32 var_7;
    float32 var_8;
    vec3 var_9;
    float32 var_10;
    bool var_11;
    bool var_12;
    bool var_13;
    // vec3 var_14;
    bool var_15;
    float32 var_16;
    float32 var_17;
    float32 var_18;
    float32 var_19;
    float32 var_20;
    float32 var_21;
    bool var_22;
    float32 var_23;
    float32 var_24;
    const float32 var_25 = 1.0;
    float32 var_26;
    float32 var_27;
    float32 var_28;
    float32 var_29;
    float32 var_30;
    float32 var_31;
    float32 var_32;
    float32 var_33;
    bool var_34;
    float32 var_35;
    float32 var_36;
    float32 var_37;
    float32 var_38;
    float32 var_39;
    float32 var_40;
    float32 var_41;
    float32 var_42;
    float32 var_43;
    float32 var_44;
    bool var_45;
    float32 var_46;
    float32 var_47;
    float32 var_48;
    float32 var_49;
    float32 var_50;
    bool var_51;
    float32 var_52;
    float32 var_53;
    float32 var_54;
    float32 var_55;
    float32 var_56;
    float32 var_57;
    float32 var_58;
    float32 var_59;
    float32 var_60;
    float32 var_61;
    float32 var_62;
    vec3 var_63;
    vec3 var_64;
    vec3 var_65;
    vec3 var_66;
    vec3 var_67;
    vec3 var_68;
    vec3 var_69;
    float32 var_70;
    // vec3 var_71;
    //---------
    // dual vars
    vec3 adj_0 = 0;
    vec3 adj_1 = 0;
    vec3 adj_2 = 0;
    float32 adj_3 = 0;
    float32 adj_4 = 0;
    float32 adj_5 = 0;
    float32 adj_6 = 0;
    float32 adj_7 = 0;
    float32 adj_8 = 0;
    vec3 adj_9 = 0;
    float32 adj_10 = 0;
    //bool adj_11 = 0;
    //bool adj_12 = 0;
    //bool adj_13 = 0;
    vec3 adj_14 = 0;
    bool adj_15 = 0;
    float32 adj_16 = 0;
    float32 adj_17 = 0;
    float32 adj_18 = 0;
    float32 adj_19 = 0;
    float32 adj_20 = 0;
    float32 adj_21 = 0;
    bool adj_22 = 0;
    float32 adj_23 = 0;
    float32 adj_24 = 0;
    float32 adj_25 = 0;
    float32 adj_26 = 0;
    float32 adj_27 = 0;
    float32 adj_28 = 0;
    float32 adj_29 = 0;
    float32 adj_30 = 0;
    float32 adj_31 = 0;
    float32 adj_32 = 0;
    float32 adj_33 = 0;
    bool adj_34 = 0;
    float32 adj_35 = 0;
    float32 adj_36 = 0;
    float32 adj_37 = 0;
    float32 adj_38 = 0;
    float32 adj_39 = 0;
    float32 adj_40 = 0;
    float32 adj_41 = 0;
    float32 adj_42 = 0;
    float32 adj_43 = 0;
    float32 adj_44 = 0;
    bool adj_45 = 0;
    float32 adj_46 = 0;
    float32 adj_47 = 0;
    float32 adj_48 = 0;
    float32 adj_49 = 0;
    float32 adj_50 = 0;
    bool adj_51 = 0;
    float32 adj_52 = 0;
    float32 adj_53 = 0;
    float32 adj_54 = 0;
    float32 adj_55 = 0;
    float32 adj_56 = 0;
    float32 adj_57 = 0;
    float32 adj_58 = 0;
    float32 adj_59 = 0;
    float32 adj_60 = 0;
    float32 adj_61 = 0;
    float32 adj_62 = 0;
    vec3 adj_63 = 0;
    vec3 adj_64 = 0;
    vec3 adj_65 = 0;
    vec3 adj_66 = 0;
    vec3 adj_67 = 0;
    vec3 adj_68 = 0;
    vec3 adj_69 = 0;
    float32 adj_70 = 0;
    vec3 adj_71 = 0;
    //---------
    // forward
    var_0 = wp::sub(var_q1, var_p1);
    var_1 = wp::sub(var_q2, var_p2);
    var_2 = wp::sub(var_p1, var_p2);
    var_3 = wp::dot(var_0, var_0);
    var_4 = wp::dot(var_1, var_1);
    var_5 = wp::dot(var_1, var_2);
    var_7 = wp::cast_float(var_6);
    var_8 = wp::cast_float(var_6);
    var_9 = wp::sub(var_p2, var_p1);
    var_10 = wp::length(var_9);
    var_11 = (var_3 <= var_epsilon);
    var_12 = (var_4 <= var_epsilon);
    var_13 = var_11 && var_12;
    if (var_13) {
    	// var_14 = wp::vec3(var_7, var_8, var_10);
    	goto label0;
    }
    var_15 = (var_3 <= var_epsilon);
    if (var_15) {
    	var_16 = wp::cast_float(var_6);
    	var_17 = wp::div(var_5, var_4);
    	var_18 = wp::cast_float(var_17);
    }
    var_19 = wp::where(var_15, var_16, var_7);
    var_20 = wp::where(var_15, var_18, var_8);
    if (!var_15) {
    	var_21 = wp::dot(var_0, var_2);
    	var_22 = (var_4 <= var_epsilon);
    	if (var_22) {
    		var_23 = wp::neg(var_21);
    		var_24 = wp::div(var_23, var_3);
    		var_26 = wp::clamp(var_24, var_6, var_25);
    		var_27 = wp::cast_float(var_6);
    	}
    	var_28 = wp::where(var_22, var_26, var_19);
    	var_29 = wp::where(var_22, var_27, var_20);
    	if (!var_22) {
    		var_30 = wp::dot(var_0, var_1);
    		var_31 = wp::mul(var_3, var_4);
    		var_32 = wp::mul(var_30, var_30);
    		var_33 = wp::sub(var_31, var_32);
    		var_34 = (var_33 != var_6);
    		if (var_34) {
    			var_35 = wp::mul(var_30, var_5);
    			var_36 = wp::mul(var_21, var_4);
    			var_37 = wp::sub(var_35, var_36);
    			var_38 = wp::div(var_37, var_33);
    			var_39 = wp::clamp(var_38, var_6, var_25);
    		}
    		var_40 = wp::where(var_34, var_39, var_28);
    		if (!var_34) {
    		}
    		var_41 = wp::where(var_34, var_40, var_6);
    		var_42 = wp::mul(var_30, var_41);
    		var_43 = wp::add(var_42, var_5);
    		var_44 = wp::div(var_43, var_4);
    		var_45 = (var_44 < var_6);
    		if (var_45) {
    			var_46 = wp::neg(var_21);
    			var_47 = wp::div(var_46, var_3);
    			var_48 = wp::clamp(var_47, var_6, var_25);
    		}
    		var_49 = wp::where(var_45, var_48, var_41);
    		var_50 = wp::where(var_45, var_6, var_44);
    		if (!var_45) {
    			var_51 = (var_50 > var_25);
    			if (var_51) {
    				var_52 = wp::sub(var_30, var_21);
    				var_53 = wp::div(var_52, var_3);
    				var_54 = wp::clamp(var_53, var_6, var_25);
    			}
    			var_55 = wp::where(var_51, var_54, var_49);
    			var_56 = wp::where(var_51, var_25, var_50);
    		}
    		var_57 = wp::where(var_45, var_49, var_55);
    		var_58 = wp::where(var_45, var_50, var_56);
    	}
    	var_59 = wp::where(var_22, var_28, var_57);
    	var_60 = wp::where(var_22, var_29, var_58);
    }
    var_61 = wp::where(var_15, var_19, var_59);
    var_62 = wp::where(var_15, var_20, var_60);
    var_63 = wp::sub(var_q1, var_p1);
    var_64 = wp::mul(var_63, var_61);
    var_65 = wp::add(var_p1, var_64);
    var_66 = wp::sub(var_q2, var_p2);
    var_67 = wp::mul(var_66, var_62);
    var_68 = wp::add(var_p2, var_67);
    var_69 = wp::sub(var_68, var_65);
    var_70 = wp::length(var_69);
    // var_71 = wp::vec3(var_61, var_62, var_70);
    goto label1;
    //---------
    // reverse
    label1:;
    adj_71 += adj_ret;
    wp::adj_vec3(var_61, var_62, var_70, adj_61, adj_62, adj_70, adj_71);
    wp::adj_length(var_69, var_70, adj_69, adj_70);
    wp::adj_sub(var_68, var_65, adj_68, adj_65, adj_69);
    wp::adj_add(var_p2, var_67, adj_p2, adj_67, adj_68);
    wp::adj_mul(var_66, var_62, adj_66, adj_62, adj_67);
    wp::adj_sub(var_q2, var_p2, adj_q2, adj_p2, adj_66);
    wp::adj_add(var_p1, var_64, adj_p1, adj_64, adj_65);
    wp::adj_mul(var_63, var_61, adj_63, adj_61, adj_64);
    wp::adj_sub(var_q1, var_p1, adj_q1, adj_p1, adj_63);
    wp::adj_where(var_15, var_20, var_60, adj_15, adj_20, adj_60, adj_62);
    wp::adj_where(var_15, var_19, var_59, adj_15, adj_19, adj_59, adj_61);
    if (!var_15) {
    	wp::adj_where(var_22, var_29, var_58, adj_22, adj_29, adj_58, adj_60);
    	wp::adj_where(var_22, var_28, var_57, adj_22, adj_28, adj_57, adj_59);
    	if (!var_22) {
    		wp::adj_where(var_45, var_50, var_56, adj_45, adj_50, adj_56, adj_58);
    		wp::adj_where(var_45, var_49, var_55, adj_45, adj_49, adj_55, adj_57);
    		if (!var_45) {
    			wp::adj_where(var_51, var_25, var_50, adj_51, adj_25, adj_50, adj_56);
    			wp::adj_where(var_51, var_54, var_49, adj_51, adj_54, adj_49, adj_55);
    			if (var_51) {
    				wp::adj_clamp(var_53, var_6, var_25, adj_53, adj_6, adj_25, adj_54);
    				wp::adj_div(var_52, var_3, var_53, adj_52, adj_3, adj_53);
    				wp::adj_sub(var_30, var_21, adj_30, adj_21, adj_52);
    			}
    		}
    		wp::adj_where(var_45, var_6, var_44, adj_45, adj_6, adj_44, adj_50);
    		wp::adj_where(var_45, var_48, var_41, adj_45, adj_48, adj_41, adj_49);
    		if (var_45) {
    			wp::adj_clamp(var_47, var_6, var_25, adj_47, adj_6, adj_25, adj_48);
    			wp::adj_div(var_46, var_3, var_47, adj_46, adj_3, adj_47);
    			wp::adj_neg(var_21, adj_21, adj_46);
    		}
    		wp::adj_div(var_43, var_4, var_44, adj_43, adj_4, adj_44);
    		wp::adj_add(var_42, var_5, adj_42, adj_5, adj_43);
    		wp::adj_mul(var_30, var_41, adj_30, adj_41, adj_42);
    		wp::adj_where(var_34, var_40, var_6, adj_34, adj_40, adj_6, adj_41);
    		if (!var_34) {
    		}
    		wp::adj_where(var_34, var_39, var_28, adj_34, adj_39, adj_28, adj_40);
    		if (var_34) {
    			wp::adj_clamp(var_38, var_6, var_25, adj_38, adj_6, adj_25, adj_39);
    			wp::adj_div(var_37, var_33, var_38, adj_37, adj_33, adj_38);
    			wp::adj_sub(var_35, var_36, adj_35, adj_36, adj_37);
    			wp::adj_mul(var_21, var_4, adj_21, adj_4, adj_36);
    			wp::adj_mul(var_30, var_5, adj_30, adj_5, adj_35);
    		}
    		wp::adj_sub(var_31, var_32, adj_31, adj_32, adj_33);
    		wp::adj_mul(var_30, var_30, adj_30, adj_30, adj_32);
    		wp::adj_mul(var_3, var_4, adj_3, adj_4, adj_31);
    		wp::adj_dot(var_0, var_1, adj_0, adj_1, adj_30);
    	}
    	wp::adj_where(var_22, var_27, var_20, adj_22, adj_27, adj_20, adj_29);
    	wp::adj_where(var_22, var_26, var_19, adj_22, adj_26, adj_19, adj_28);
    	if (var_22) {
    		wp::adj_cast_float(var_6, adj_6, adj_27);
    		wp::adj_clamp(var_24, var_6, var_25, adj_24, adj_6, adj_25, adj_26);
    		wp::adj_div(var_23, var_3, var_24, adj_23, adj_3, adj_24);
    		wp::adj_neg(var_21, adj_21, adj_23);
    	}
    	wp::adj_dot(var_0, var_2, adj_0, adj_2, adj_21);
    }
    wp::adj_where(var_15, var_18, var_8, adj_15, adj_18, adj_8, adj_20);
    wp::adj_where(var_15, var_16, var_7, adj_15, adj_16, adj_7, adj_19);
    if (var_15) {
    	wp::adj_cast_float(var_17, adj_17, adj_18);
    	wp::adj_div(var_5, var_4, var_17, adj_5, adj_4, adj_17);
    	wp::adj_cast_float(var_6, adj_6, adj_16);
    }
    if (var_13) {
    	label0:;
    	adj_14 += adj_ret;
    	wp::adj_vec3(var_7, var_8, var_10, adj_7, adj_8, adj_10, adj_14);
    }
    wp::adj_length(var_9, var_10, adj_9, adj_10);
    wp::adj_sub(var_p2, var_p1, adj_p2, adj_p1, adj_9);
    wp::adj_cast_float(var_6, adj_6, adj_8);
    wp::adj_cast_float(var_6, adj_6, adj_7);
    wp::adj_dot(var_1, var_2, adj_1, adj_2, adj_5);
    wp::adj_dot(var_1, var_1, adj_1, adj_1, adj_4);
    wp::adj_dot(var_0, var_0, adj_0, adj_0, adj_3);
    wp::adj_sub(var_p1, var_p2, adj_p1, adj_p2, adj_2);
    wp::adj_sub(var_q2, var_p2, adj_q2, adj_p2, adj_1);
    wp::adj_sub(var_q1, var_p1, adj_q1, adj_p1, adj_0);
    return;

}

} // namespace wp
