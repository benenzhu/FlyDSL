	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0
	.p2align	8
	.type	hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0,@function
hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0:
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
	.file	1 "/root/FlyDSL/kernels" "tensor_shim.py"
	s_load_dword s12, s[0:1], 0x60	;.loc	1 289 24 prologue_end
	s_load_dwordx2 s[4:5], s[0:1], 0x18
	s_load_dwordx2 s[8:9], s[0:1], 0x30
	.file	2 "/root/FlyDSL/kernels" "hgemm_splitk.py"
	s_ashr_i32 s14, s2, 31	;.loc	2 382 0
	s_lshr_b32 s14, s14, 27
	s_add_i32 s14, s2, s14
	s_waitcnt lgkmcnt(0)	;.loc	1 289 24
	s_ashr_i32 s13, s12, 31
	s_and_b32 s5, s5, 0xffff
	s_and_b32 s9, s9, 0xffff
	s_ashr_i32 s18, s14, 5	;.loc	2 382 0
	s_and_b32 s19, s14, 0xffffffe0
	s_cmp_lg_u32 s2, s19
	s_cselect_b64 s[14:15], -1, 0
	s_cmp_lt_i32 s2, 0
	s_cselect_b64 s[16:17], -1, 0
	s_and_b64 s[14:15], s[16:17], s[14:15]
	v_lshrrev_b32_e32 v002, 6, v000	;.loc	2 375 0
	s_subb_u32 s14, s18, 0	;.loc	2 382 0
	s_sub_i32 s15, s2, s19
	s_lshl_b32 s18, s14, 8	;.loc	2 388 0
	v_readfirstlane_b32 s14, v002	;.loc	2 552 0
	s_lshl_b32 s16, s15, 8	;.loc	2 389 0
	s_lshl_b32 s19, s14, 10	;.loc	2 552 0
	s_mul_i32 s2, s3, 0x3fc0	;.loc	2 386 0
	.file	3 "/root/FlyDSL/python/flydsl/expr" "numeric.py"
	s_ashr_i32 s17, s18, 31	;.loc	3 875 16
	s_ashr_i32 s3, s16, 31
	s_add_i32 s24, s19, 0x10000	;.loc	2 583 0
	v_lshrrev_b32_e32 v002, 2, v000	;.loc	2 626 0
	s_cmpk_lt_u32 s16, 0x2000	;.loc	2 644 12
	v_xor_b32_e32 v003, v002, v000	;.loc	2 630 0
	s_cselect_b64 vcc, -1, 0	;.loc	2 644 12
	v_lshlrev_b32_e32 v003, 3, v003	;.loc	2 630 0
	s_and_b64 s[14:15], vcc, exec	;.loc	2 643 23
	v_and_b32_e32 v105, 24, v003	;.loc	2 630 0
	v_or_b32_e32 v003, s16, v002	;.loc	2 642 0
	s_cselect_b32 s3, s3, 0	;.loc	2 643 23
	v_cndmask_b32_e32 v004, 0, v003, vcc
	v_mov_b32_e32 v003, s3
	v_alignbit_b32 v003, v003, v004, 8	;.loc	2 355 0
	s_movk_i32 s28, 0xff
	v_mad_u64_u32 v098 v099, s[14:15], v003, s28, 0
	v_lshlrev_b32_e32 v003, 6, v004
	s_lshr_b32 s29, s2, 6	;.loc	2 353 0
	v_and_b32_e32 v100, 0xfc0, v003	;.loc	2 355 0
	v_add_u32_e32 v003, s29, v098
	v_lshl_or_b32 v004, v003, 14, v100
	v_or_b32_e32 v003, v004, v105
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, -1
	v_lshlrev_b32_e32 v003, 1, v003	;.loc	2 652 24
	v_or_b32_e32 v005, 64, v002	;.loc	2 628 0
	s_mov_b32 s10, s6	;.loc	1 289 24
	s_mov_b32 s11, s7
	s_mov_b32 m0, s24	;.loc	2 567 0
	buffer_load_dwordx4 v003, s[8:11], 0 offen sc0 lds
	v_or_b32_e32 v003, s16, v005	;.loc	2 642 0
	v_cndmask_b32_e32 v006, 0, v003, vcc	;.loc	2 643 23
	v_mov_b32_e32 v003, s3
	v_alignbit_b32 v003, v003, v006, 8	;.loc	2 355 0
	v_mad_u64_u32 v102 v103, s[14:15], v003, s28, 0
	v_lshlrev_b32_e32 v003, 6, v006
	v_and_b32_e32 v104, 0x3fc0, v003
	v_add_u32_e32 v003, s29, v102
	v_lshl_or_b32 v006, v003, 14, v104
	v_or_b32_e32 v003, v006, v105
	v_lshlrev_b32_e32 v003, 1, v003	;.loc	2 652 24
	v_or_b32_e32 v007, 0x80, v002	;.loc	2 628 0
	s_add_i32 s23, s19, 0x11000	;.loc	2 583 0
	s_mov_b32 m0, s23	;.loc	2 567 0
	buffer_load_dwordx4 v003, s[8:11], 0 offen sc0 lds
	v_or_b32_e32 v003, s16, v007	;.loc	2 642 0
	v_cndmask_b32_e32 v008, 0, v003, vcc	;.loc	2 643 23
	v_mov_b32_e32 v003, s3
	v_alignbit_b32 v003, v003, v008, 8	;.loc	2 355 0
	v_mad_u64_u32 v106 v107, s[14:15], v003, s28, 0
	v_lshlrev_b32_e32 v003, 6, v008
	v_and_b32_e32 v108, 0x3fc0, v003
	v_add_u32_e32 v003, s29, v106
	v_lshl_or_b32 v008, v003, 14, v108
	v_or_b32_e32 v003, v008, v105
	v_lshlrev_b32_e32 v003, 1, v003	;.loc	2 652 24
	v_or_b32_e32 v009, 0xc0, v002	;.loc	2 628 0
	s_add_i32 s22, s19, 0x12000	;.loc	2 583 0
	s_mov_b32 m0, s22	;.loc	2 567 0
	buffer_load_dwordx4 v003, s[8:11], 0 offen sc0 lds
	v_or_b32_e32 v003, s16, v009	;.loc	2 642 0
	v_cndmask_b32_e32 v010, 0, v003, vcc	;.loc	2 643 23
	v_mov_b32_e32 v003, s3
	v_alignbit_b32 v003, v003, v010, 8	;.loc	2 355 0
	v_mad_u64_u32 v110 v111, s[14:15], v003, s28, 0
	v_lshlrev_b32_e32 v003, 6, v010
	v_and_b32_e32 v112, 0x3fc0, v003
	v_add_u32_e32 v003, s29, v110
	v_lshl_or_b32 v003, v003, 14, v112
	v_or_b32_e32 v010, v003, v105
	v_lshlrev_b32_e32 v010, 1, v010	;.loc	2 652 24
	s_add_i32 s21, s19, 0x13000	;.loc	2 583 0
	s_mov_b32 m0, s21	;.loc	2 567 0
	buffer_load_dwordx4 v010, s[8:11], 0 offen sc0 lds
	v_or_b32_e32 v010, 32, v105	;.loc	2 631 0
	v_or_b32_e32 v004, v004, v010	;.loc	2 355 0
	v_lshlrev_b32_e32 v004, 1, v004	;.loc	2 652 24
	s_add_i32 s3, s19, 0x14000	;.loc	2 583 0
	s_mov_b32 m0, s3	;.loc	2 567 0
	buffer_load_dwordx4 v004, s[8:11], 0 offen sc0 lds
	v_or_b32_e32 v004, v006, v010	;.loc	2 355 0
	s_add_i32 s3, s19, 0x15000	;.loc	2 583 0
	v_lshlrev_b32_e32 v004, 1, v004	;.loc	2 652 24
	v_or_b32_e32 v003, v003, v010	;.loc	2 355 0
	s_mov_b32 m0, s3	;.loc	2 567 0
	buffer_load_dwordx4 v004, s[8:11], 0 offen sc0 lds
	s_add_i32 s3, s19, 0x16000	;.loc	2 583 0
	v_or_b32_e32 v004, v008, v010	;.loc	2 355 0
	v_lshlrev_b32_e32 v003, 1, v003	;.loc	2 652 24
	v_lshlrev_b32_e32 v004, 1, v004
	s_mov_b32 m0, s3	;.loc	2 567 0
	buffer_load_dwordx4 v004, s[8:11], 0 offen sc0 lds
	s_add_i32 s3, s19, 0x17000	;.loc	2 583 0
	s_mov_b32 m0, s3	;.loc	2 567 0
	buffer_load_dwordx4 v003, s[8:11], 0 offen sc0 lds
	v_or_b32_e32 v002, s18, v002	;.loc	2 605 0
	v_mov_b32_e32 v003, s17
	v_mov_b32_e32 v004, s17	;.loc	2 606 23
	v_cmp_gt_u64_e32 vcc, s[12:13], v002 v003	;.loc	2 607 12
	s_add_i32 s3, s19, 0x3000	;.loc	2 583 0
	v_and_b32_e32 v001, 0x80, v000	;.loc	2 392 0
	v_cndmask_b32_e32 v003, 0, v004, vcc	;.loc	2 606 23
	v_cndmask_b32_e32 v002, 0, v002, vcc
	v_alignbit_b32 v003, v003, v002, 8	;.loc	2 355 0
	v_mad_u64_u32 v114 v115, s[14:15], v003, s28, 0
	v_lshlrev_b32_e32 v002, 6, v002
	v_and_b32_e32 v116, 0xfc0, v002
	v_add_u32_e32 v002, s29, v114
	v_lshl_or_b32 v006, v002, 14, v116
	v_or_b32_e32 v002, v006, v105
	v_lshlrev_b32_e32 v002, 1, v002	;.loc	2 615 24
	s_mov_b32 m0, s19	;.loc	2 567 0
	buffer_load_dwordx4 v002, s[4:7], 0 offen sc0 lds
	v_or_b32_e32 v002, s18, v005	;.loc	2 605 0
	v_mov_b32_e32 v003, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v002 v003	;.loc	2 607 12
	s_add_i32 s15, s19, 0x1000	;.loc	2 583 0
	s_add_i32 s14, s19, 0x2000
	v_cndmask_b32_e32 v003, 0, v004, vcc	;.loc	2 606 23
	v_cndmask_b32_e32 v002, 0, v002, vcc
	v_alignbit_b32 v003, v003, v002, 8	;.loc	2 355 0
	v_mad_u64_u32 v118 v119, s[26:27], v003, s28, 0
	v_lshlrev_b32_e32 v002, 6, v002
	v_and_b32_e32 v120, 0x3fc0, v002
	v_add_u32_e32 v002, s29, v118
	v_lshl_or_b32 v005, v002, 14, v120
	v_or_b32_e32 v002, v005, v105
	v_lshlrev_b32_e32 v002, 1, v002	;.loc	2 615 24
	s_mov_b32 m0, s15	;.loc	2 567 0
	buffer_load_dwordx4 v002, s[4:7], 0 offen sc0 lds
	v_or_b32_e32 v002, s18, v007	;.loc	2 605 0
	v_mov_b32_e32 v003, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v002 v003	;.loc	2 607 12
	v_lshlrev_b32_e32 v034, 1, v000	;.loc	2 393 0
	v_and_b32_e32 v035, 15, v000	;.loc	2 394 0
	v_cndmask_b32_e32 v003, 0, v004, vcc	;.loc	2 606 23
	v_cndmask_b32_e32 v002, 0, v002, vcc
	v_alignbit_b32 v003, v003, v002, 8	;.loc	2 355 0
	v_mad_u64_u32 v122 v123, s[26:27], v003, s28, 0
	v_lshlrev_b32_e32 v002, 6, v002
	v_and_b32_e32 v124, 0x3fc0, v002
	v_add_u32_e32 v002, s29, v122
	v_lshl_or_b32 v007, v002, 14, v124
	v_or_b32_e32 v002, v007, v105
	v_lshlrev_b32_e32 v002, 1, v002	;.loc	2 615 24
	s_mov_b32 m0, s14	;.loc	2 567 0
	buffer_load_dwordx4 v002, s[4:7], 0 offen sc0 lds
	v_or_b32_e32 v002, s18, v009	;.loc	2 605 0
	v_mov_b32_e32 v003, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v002 v003	;.loc	2 607 12
	v_lshrrev_b32_e32 v101, 4, v000	;.loc	2 395 0
	s_mov_b32 s20, 0x10000
	v_cndmask_b32_e32 v003, 0, v004, vcc	;.loc	2 606 23
	v_cndmask_b32_e32 v002, 0, v002, vcc
	v_alignbit_b32 v003, v003, v002, 8	;.loc	2 355 0
	v_mad_u64_u32 v126 v127, s[26:27], v003, s28, 0
	v_lshlrev_b32_e32 v002, 6, v002
	v_and_b32_e32 v128, 0x3fc0, v002
	v_add_u32_e32 v002, s29, v126
	v_lshl_or_b32 v002, v002, 14, v128
	v_or_b32_e32 v003, v002, v105
	v_lshlrev_b32_e32 v003, 1, v003	;.loc	2 615 24
	s_mov_b32 m0, s3	;.loc	2 567 0
	buffer_load_dwordx4 v003, s[4:7], 0 offen sc0 lds
	v_or_b32_e32 v003, v006, v010	;.loc	2 355 0
	v_lshlrev_b32_e32 v003, 1, v003	;.loc	2 615 24
	s_add_i32 s26, s19, 0x4000	;.loc	2 583 0
	s_mov_b32 m0, s26	;.loc	2 567 0
	buffer_load_dwordx4 v003, s[4:7], 0 offen sc0 lds
	v_or_b32_e32 v003, v005, v010	;.loc	2 355 0
	s_add_i32 s26, s19, 0x5000	;.loc	2 583 0
	v_lshlrev_b32_e32 v003, 1, v003	;.loc	2 615 24
	s_mov_b32 m0, s26	;.loc	2 567 0
	buffer_load_dwordx4 v003, s[4:7], 0 offen sc0 lds
	s_add_i32 s26, s19, 0x6000	;.loc	2 583 0
	v_or_b32_e32 v003, v007, v010	;.loc	2 355 0
	v_lshlrev_b32_e32 v003, 1, v003	;.loc	2 615 24
	s_mov_b32 m0, s26	;.loc	2 567 0
	buffer_load_dwordx4 v003, s[4:7], 0 offen sc0 lds
	s_add_i32 s26, s19, 0x7000	;.loc	2 583 0
	v_or_b32_e32 v002, v002, v010	;.loc	2 355 0
	v_lshlrev_b32_e32 v002, 1, v002	;.loc	2 615 24
	s_mov_b32 m0, s26	;.loc	2 567 0
	buffer_load_dwordx4 v002, s[4:7], 0 offen sc0 lds
	s_add_i32 s26, s2, 64	;.loc	2 1046 0
	s_lshr_b32 s26, s26, 6	;.loc	2 353 0
	v_add_lshl_u32 v002, v098, s26, 14	;.loc	2 355 0
	v_or3_b32 v002, v002, v100, v105
	v_lshlrev_b32_e32 v002, 1, v002	;.loc	2 652 24
	s_add_i32 s27, s19, 0x18000	;.loc	2 583 0
	s_mov_b32 m0, s27	;.loc	2 567 0
	buffer_load_dwordx4 v002, s[8:11], 0 offen sc0 lds
	v_add_lshl_u32 v002, v102, s26, 14	;.loc	2 355 0
	v_or3_b32 v002, v002, v104, v105
	v_lshlrev_b32_e32 v002, 1, v002	;.loc	2 652 24
	s_add_i32 s27, s19, 0x19000	;.loc	2 583 0
	s_mov_b32 m0, s27	;.loc	2 567 0
	buffer_load_dwordx4 v002, s[8:11], 0 offen sc0 lds
	v_add_lshl_u32 v002, v106, s26, 14	;.loc	2 355 0
	v_or3_b32 v002, v002, v108, v105
	v_lshlrev_b32_e32 v002, 1, v002	;.loc	2 652 24
	s_add_i32 s27, s19, 0x1a000	;.loc	2 583 0
	s_mov_b32 m0, s27	;.loc	2 567 0
	buffer_load_dwordx4 v002, s[8:11], 0 offen sc0 lds
	v_add_lshl_u32 v002, v110, s26, 14	;.loc	2 355 0
	v_or3_b32 v002, v002, v112, v105
	v_lshlrev_b32_e32 v002, 1, v002	;.loc	2 652 24
	s_add_i32 s27, s19, 0x1b000	;.loc	2 583 0
	s_mov_b32 m0, s27	;.loc	2 567 0
	buffer_load_dwordx4 v002, s[8:11], 0 offen sc0 lds
	v_add_lshl_u32 v002, v114, s26, 14	;.loc	2 355 0
	v_or3_b32 v002, v002, v116, v105
	v_lshlrev_b32_e32 v002, 1, v002	;.loc	2 615 24
	s_add_i32 s27, s19, 0x8000	;.loc	2 583 0
	s_mov_b32 m0, s27	;.loc	2 567 0
	buffer_load_dwordx4 v002, s[4:7], 0 offen sc0 lds
	v_add_lshl_u32 v002, v118, s26, 14	;.loc	2 355 0
	v_or3_b32 v002, v002, v120, v105
	v_lshlrev_b32_e32 v002, 1, v002	;.loc	2 615 24
	s_add_i32 s27, s19, 0x9000	;.loc	2 583 0
	s_mov_b32 m0, s27	;.loc	2 567 0
	buffer_load_dwordx4 v002, s[4:7], 0 offen sc0 lds
	v_add_lshl_u32 v002, v122, s26, 14	;.loc	2 355 0
	v_or3_b32 v002, v002, v124, v105
	v_lshlrev_b32_e32 v002, 1, v002	;.loc	2 615 24
	s_add_i32 s27, s19, 0xa000	;.loc	2 583 0
	s_mov_b32 m0, s27	;.loc	2 567 0
	buffer_load_dwordx4 v002, s[4:7], 0 offen sc0 lds
	v_add_lshl_u32 v002, v126, s26, 14	;.loc	2 355 0
	v_or3_b32 v002, v002, v128, v105
	v_lshlrev_b32_e32 v002, 1, v002	;.loc	2 615 24
	s_movk_i32 s25, 0x80
	v_or_b32_e32 v113, s2, v105	;.loc	2 631 0
	s_add_i32 s27, s19, 0xb000	;.loc	2 583 0
	s_mov_b32 m0, s27	;.loc	2 567 0
	buffer_load_dwordx4 v002, s[4:7], 0 offen sc0 lds
	v_xor_b32_e32 v002, v101, v000	;.loc	2 701 0
	v_lshlrev_b32_e32 v002, 4, v002
	v_and_b32_e32 v103, 48, v002
	v_lshlrev_b32_e32 v002, 5, v000	;.loc	2 702 0
	v_and_b32_e32 v002, 0x11e0, v002
	v_and_or_b32 v099, v034, s25, v035	;.loc	2 713 0
	s_add_i32 s25, s2, 0x60	;.loc	2 631 0
	v_lshlrev_b32_e32 v111, 1, v002	;.loc	1 317 12
	v_or3_b32 v006, v001, v035, 16	;.loc	2 696 0
	v_mov_b32_e32 v034, 0x10000	;.loc	1 317 12
	s_lshr_b32 s28, s25, 6	;.loc	2 353 0
	v_or_b32_e32 v117, v111, v103	;.loc	1 317 12
	v_lshlrev_b32_e32 v109, 6, v006
	v_lshl_or_b32 v107, v099, 6, v034
	v_bitop3_b32 v090, v105, 56, s25 bitop3:0xc8	;.loc	2 354 0
	v_add_lshl_u32 v034, v098, s28, 14	;.loc	2 355 0
	s_waitcnt vmcnt(0)	;.loc	2 408 0
	s_barrier
	ds_read_b128 v002 v003 v004 v005, v117	;.loc	1 317 12
	v_or_b32_e32 v119, v109, v103
	v_or_b32_e32 v121, v107, v103
	v_or3_b32 v034, v034, v100, v090	;.loc	2 355 0
	v_lshlrev_b32_e32 v115, 5, v006	;.loc	2 702 0
	ds_read_b128 v006 v007 v008 v009, v119	;.loc	1 317 12
	ds_read_b128 v010 v011 v012 v013, v119 offset:1024
	ds_read_b128 v014 v015 v016 v017, v119 offset:2048
	ds_read_b128 v018 v019 v020 v021, v119 offset:3072
	ds_read_b128 v022 v023 v024 v025, v119 offset:4096
	ds_read_b128 v026 v027 v028 v029, v119 offset:5120
	ds_read_b128 v030 v031 v032 v033, v119 offset:6144
	ds_read_b128 v038 v039 v040 v041, v121
	ds_read_b128 v042 v043 v044 v045, v121 offset:1024
	ds_read_b128 v046 v047 v048 v049, v121 offset:2048
	ds_read_b128 v054 v055 v056 v057, v121 offset:3072
	ds_read_b128 v094 v095 v096 v097, v121 offset:4096
	ds_read_b128 v130 v131 v132 v133, v121 offset:5120
	ds_read_b128 v134 v135 v136 v137, v121 offset:6144
	ds_read_b128 v138 v139 v140 v141, v121 offset:7168
	s_waitcnt lgkmcnt(7)	;.loc	2 977 0
	v_mfma_f32_16x16x32_bf16 a[252:255], v002 v003 v004 v005, v038 v039 v040 v041, 0
	s_add_i32 s27, s19, 0x1c000	;.loc	2 583 0
	v_lshlrev_b32_e32 v034, 1, v034	;.loc	2 652 24
	s_mov_b32 m0, s27	;.loc	2 567 0
	buffer_load_dwordx4 v034, s[8:11], 0 offen sc0 lds
	ds_read_b128 v142 v143 v144 v145, v117 offset:16384	;.loc	1 317 12
	s_waitcnt lgkmcnt(7)	;.loc	2 977 0
	v_mfma_f32_16x16x32_bf16 a[248:251], v002 v003 v004 v005, v042 v043 v044 v045, 0
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 a[244:247], v002 v003 v004 v005, v046 v047 v048 v049, 0
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 a[240:243], v002 v003 v004 v005, v054 v055 v056 v057, 0
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 a[236:239], v002 v003 v004 v005, v094 v095 v096 v097, 0
	ds_read_b128 v146 v147 v148 v149, v119 offset:16384	;.loc	1 317 12
	s_waitcnt lgkmcnt(4)	;.loc	2 977 0
	v_mfma_f32_16x16x32_bf16 a[232:235], v002 v003 v004 v005, v130 v131 v132 v133, 0
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 a[228:231], v002 v003 v004 v005, v134 v135 v136 v137, 0
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 a[224:227], v002 v003 v004 v005, v138 v139 v140 v141, 0
	v_add_lshl_u32 v002, v102, s28, 14	;.loc	2 355 0
	v_or3_b32 v002, v002, v104, v090
	v_lshlrev_b32_e32 v002, 1, v002	;.loc	2 652 24
	v_mfma_f32_16x16x32_bf16 a[220:223], v006 v007 v008 v009, v038 v039 v040 v041, 0	;.loc	2 977 0
	s_add_i32 s25, s19, 0x1d000	;.loc	2 583 0
	s_mov_b32 m0, s25	;.loc	2 567 0
	buffer_load_dwordx4 v002, s[8:11], 0 offen sc0 lds
	ds_read_b128 v002 v003 v004 v005, v119 offset:17408	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[216:219], v006 v007 v008 v009, v042 v043 v044 v045, 0	;.loc	2 977 0
	v_mfma_f32_16x16x32_bf16 a[212:215], v006 v007 v008 v009, v046 v047 v048 v049, 0
	v_mfma_f32_16x16x32_bf16 a[208:211], v006 v007 v008 v009, v054 v055 v056 v057, 0
	v_mfma_f32_16x16x32_bf16 a[204:207], v006 v007 v008 v009, v094 v095 v096 v097, 0
	ds_read_b128 v150 v151 v152 v153, v119 offset:18432	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[200:203], v006 v007 v008 v009, v130 v131 v132 v133, 0	;.loc	2 977 0
	v_mfma_f32_16x16x32_bf16 a[196:199], v006 v007 v008 v009, v134 v135 v136 v137, 0
	v_mfma_f32_16x16x32_bf16 a[192:195], v006 v007 v008 v009, v138 v139 v140 v141, 0
	v_add_lshl_u32 v006, v106, s28, 14	;.loc	2 355 0
	v_or3_b32 v006, v006, v108, v090
	v_lshlrev_b32_e32 v006, 1, v006	;.loc	2 652 24
	v_mfma_f32_16x16x32_bf16 a[188:191], v010 v011 v012 v013, v038 v039 v040 v041, 0	;.loc	2 977 0
	s_add_i32 s25, s19, 0x1e000	;.loc	2 583 0
	s_mov_b32 m0, s25	;.loc	2 567 0
	buffer_load_dwordx4 v006, s[8:11], 0 offen sc0 lds
	v_add_lshl_u32 v006, v110, s28, 14	;.loc	2 355 0
	v_or3_b32 v006, v006, v112, v090
	v_lshlrev_b32_e32 v006, 1, v006	;.loc	2 652 24
	ds_read_b128 v034 v035 v036 v037, v119 offset:19456	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[184:187], v010 v011 v012 v013, v042 v043 v044 v045, 0	;.loc	2 977 0
	v_mfma_f32_16x16x32_bf16 a[180:183], v010 v011 v012 v013, v046 v047 v048 v049, 0
	v_mfma_f32_16x16x32_bf16 a[176:179], v010 v011 v012 v013, v054 v055 v056 v057, 0
	v_mfma_f32_16x16x32_bf16 a[172:175], v010 v011 v012 v013, v094 v095 v096 v097, 0
	ds_read_b128 v050 v051 v052 v053, v119 offset:20480	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[168:171], v010 v011 v012 v013, v130 v131 v132 v133, 0	;.loc	2 977 0
	v_mfma_f32_16x16x32_bf16 a[164:167], v010 v011 v012 v013, v134 v135 v136 v137, 0
	v_mfma_f32_16x16x32_bf16 a[160:163], v010 v011 v012 v013, v138 v139 v140 v141, 0
	v_mfma_f32_16x16x32_bf16 a[156:159], v014 v015 v016 v017, v038 v039 v040 v041, 0
	s_add_i32 s25, s19, 0x1f000	;.loc	2 583 0
	s_mov_b32 m0, s25	;.loc	2 567 0
	buffer_load_dwordx4 v006, s[8:11], 0 offen sc0 lds
	v_add_lshl_u32 v006, v114, s28, 14	;.loc	2 355 0
	v_or3_b32 v006, v006, v116, v090
	v_lshlrev_b32_e32 v006, 1, v006	;.loc	2 615 24
	ds_read_b128 v058 v059 v060 v061, v119 offset:21504	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[152:155], v014 v015 v016 v017, v042 v043 v044 v045, 0	;.loc	2 977 0
	v_mfma_f32_16x16x32_bf16 a[148:151], v014 v015 v016 v017, v046 v047 v048 v049, 0
	v_mfma_f32_16x16x32_bf16 a[144:147], v014 v015 v016 v017, v054 v055 v056 v057, 0
	v_mfma_f32_16x16x32_bf16 a[140:143], v014 v015 v016 v017, v094 v095 v096 v097, 0
	ds_read_b128 v066 v067 v068 v069, v119 offset:22528	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[136:139], v014 v015 v016 v017, v130 v131 v132 v133, 0	;.loc	2 977 0
	v_mfma_f32_16x16x32_bf16 a[132:135], v014 v015 v016 v017, v134 v135 v136 v137, 0
	v_mfma_f32_16x16x32_bf16 a[128:131], v014 v015 v016 v017, v138 v139 v140 v141, 0
	v_mfma_f32_16x16x32_bf16 a[124:127], v018 v019 v020 v021, v038 v039 v040 v041, 0
	s_add_i32 s25, s19, 0xc000	;.loc	2 583 0
	s_mov_b32 m0, s25	;.loc	2 567 0
	buffer_load_dwordx4 v006, s[4:7], 0 offen sc0 lds
	v_add_lshl_u32 v006, v118, s28, 14	;.loc	2 355 0
	v_or3_b32 v006, v006, v120, v090
	v_lshlrev_b32_e32 v006, 1, v006	;.loc	2 615 24
	ds_read_b128 v070 v071 v072 v073, v121 offset:16384	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[120:123], v018 v019 v020 v021, v042 v043 v044 v045, 0	;.loc	2 977 0
	v_mfma_f32_16x16x32_bf16 a[116:119], v018 v019 v020 v021, v046 v047 v048 v049, 0
	v_mfma_f32_16x16x32_bf16 a[112:115], v018 v019 v020 v021, v054 v055 v056 v057, 0
	v_mfma_f32_16x16x32_bf16 a[108:111], v018 v019 v020 v021, v094 v095 v096 v097, 0
	ds_read_b128 v062 v063 v064 v065, v121 offset:17408	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[104:107], v018 v019 v020 v021, v130 v131 v132 v133, 0	;.loc	2 977 0
	v_mfma_f32_16x16x32_bf16 a[100:103], v018 v019 v020 v021, v134 v135 v136 v137, 0
	v_mfma_f32_16x16x32_bf16 a[96:99], v018 v019 v020 v021, v138 v139 v140 v141, 0
	v_mfma_f32_16x16x32_bf16 a[92:95], v022 v023 v024 v025, v038 v039 v040 v041, 0
	s_add_i32 s25, s19, 0xd000	;.loc	2 583 0
	s_mov_b32 m0, s25	;.loc	2 567 0
	buffer_load_dwordx4 v006, s[4:7], 0 offen sc0 lds
	v_add_lshl_u32 v006, v122, s28, 14	;.loc	2 355 0
	v_or3_b32 v006, v006, v124, v090
	v_lshlrev_b32_e32 v006, 1, v006	;.loc	2 615 24
	ds_read_b128 v074 v075 v076 v077, v121 offset:18432	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[88:91], v022 v023 v024 v025, v042 v043 v044 v045, 0	;.loc	2 977 0
	v_mfma_f32_16x16x32_bf16 a[84:87], v022 v023 v024 v025, v046 v047 v048 v049, 0
	v_mfma_f32_16x16x32_bf16 a[80:83], v022 v023 v024 v025, v054 v055 v056 v057, 0
	v_mfma_f32_16x16x32_bf16 a[76:79], v022 v023 v024 v025, v094 v095 v096 v097, 0
	ds_read_b128 v078 v079 v080 v081, v121 offset:19456	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[72:75], v022 v023 v024 v025, v130 v131 v132 v133, 0	;.loc	2 977 0
	v_mfma_f32_16x16x32_bf16 a[68:71], v022 v023 v024 v025, v134 v135 v136 v137, 0
	v_mfma_f32_16x16x32_bf16 a[64:67], v022 v023 v024 v025, v138 v139 v140 v141, 0
	v_mfma_f32_16x16x32_bf16 a[60:63], v026 v027 v028 v029, v038 v039 v040 v041, 0
	s_add_i32 s25, s19, 0xe000	;.loc	2 583 0
	s_mov_b32 m0, s25	;.loc	2 567 0
	buffer_load_dwordx4 v006, s[4:7], 0 offen sc0 lds
	v_add_lshl_u32 v006, v126, s28, 14	;.loc	2 355 0
	s_add_i32 s26, s2, 0x80	;.loc	2 1124 0
	s_add_i32 s25, s19, 0xf000	;.loc	2 583 0
	v_or3_b32 v006, v006, v128, v090	;.loc	2 355 0
	ds_read_b128 v082 v083 v084 v085, v121 offset:20480	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[56:59], v026 v027 v028 v029, v042 v043 v044 v045, 0	;.loc	2 977 0
	v_mfma_f32_16x16x32_bf16 a[52:55], v026 v027 v028 v029, v046 v047 v048 v049, 0
	v_mfma_f32_16x16x32_bf16 a[48:51], v026 v027 v028 v029, v054 v055 v056 v057, 0
	v_mfma_f32_16x16x32_bf16 a[44:47], v026 v027 v028 v029, v094 v095 v096 v097, 0
	ds_read_b128 v086 v087 v088 v089, v121 offset:21504	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[40:43], v026 v027 v028 v029, v130 v131 v132 v133, 0	;.loc	2 977 0
	v_mfma_f32_16x16x32_bf16 a[36:39], v026 v027 v028 v029, v134 v135 v136 v137, 0
	v_mfma_f32_16x16x32_bf16 a[32:35], v026 v027 v028 v029, v138 v139 v140 v141, 0
	v_mfma_f32_16x16x32_bf16 a[28:31], v030 v031 v032 v033, v038 v039 v040 v041, 0
	v_lshlrev_b32_e32 v006, 1, v006	;.loc	2 615 24
	s_mov_b32 m0, s25	;.loc	2 567 0
	buffer_load_dwordx4 v006, s[4:7], 0 offen sc0 lds
	s_lshr_b32 s25, s26, 6	;.loc	2 353 0
	v_add_lshl_u32 v006, v098, s25, 14	;.loc	2 355 0
	v_or3_b32 v006, v006, v100, v105
	v_lshlrev_b32_e32 v006, 1, v006	;.loc	2 652 24
	ds_read_b128 v090 v091 v092 v093, v121 offset:22528	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[24:27], v030 v031 v032 v033, v042 v043 v044 v045, 0	;.loc	2 977 0
	v_mfma_f32_16x16x32_bf16 a[20:23], v030 v031 v032 v033, v046 v047 v048 v049, 0
	v_mfma_f32_16x16x32_bf16 a[16:19], v030 v031 v032 v033, v054 v055 v056 v057, 0
	v_mfma_f32_16x16x32_bf16 a[12:15], v030 v031 v032 v033, v094 v095 v096 v097, 0
	ds_read_b128 v094 v095 v096 v097, v121 offset:23552	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[8:11], v030 v031 v032 v033, v130 v131 v132 v133, 0	;.loc	2 977 0
	v_mfma_f32_16x16x32_bf16 a[4:7], v030 v031 v032 v033, v134 v135 v136 v137, 0
	v_mfma_f32_16x16x32_bf16 a[0:3], v030 v031 v032 v033, v138 v139 v140 v141, 0
	s_waitcnt vmcnt(8) lgkmcnt(0)	;.loc	2 421 0
	s_barrier
	s_waitcnt lgkmcnt(7)	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[252:255], v142 v143 v144 v145, v070 v071 v072 v073, a[252:255]
	s_mov_b32 m0, s24	;.loc	2 567 0
	buffer_load_dwordx4 v006, s[8:11], 0 offen sc0 lds
	v_add_lshl_u32 v006, v102, s25, 14	;.loc	2 355 0
	v_or3_b32 v006, v006, v104, v105
	v_lshlrev_b32_e32 v006, 1, v006	;.loc	2 652 24
	ds_read_b128 v046 v047 v048 v049, v117 offset:32768	;.loc	1 317 12
	s_waitcnt lgkmcnt(7)	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[248:251], v142 v143 v144 v145, v062 v063 v064 v065, a[248:251]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 a[244:247], v142 v143 v144 v145, v074 v075 v076 v077, a[244:247]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 a[240:243], v142 v143 v144 v145, v078 v079 v080 v081, a[240:243]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 a[236:239], v142 v143 v144 v145, v082 v083 v084 v085, a[236:239]
	ds_read_b128 v038 v039 v040 v041, v119 offset:32768	;.loc	1 317 12
	s_waitcnt lgkmcnt(4)	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[232:235], v142 v143 v144 v145, v086 v087 v088 v089, a[232:235]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 a[228:231], v142 v143 v144 v145, v090 v091 v092 v093, a[228:231]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 a[224:227], v142 v143 v144 v145, v094 v095 v096 v097, a[224:227]
	v_mfma_f32_16x16x32_bf16 a[220:223], v146 v147 v148 v149, v070 v071 v072 v073, a[220:223]
	s_mov_b32 m0, s23	;.loc	2 567 0
	buffer_load_dwordx4 v006, s[8:11], 0 offen sc0 lds
	v_add_lshl_u32 v006, v106, s25, 14	;.loc	2 355 0
	v_or3_b32 v006, v006, v108, v105
	ds_read_b128 v030 v031 v032 v033, v119 offset:33792	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[216:219], v146 v147 v148 v149, v062 v063 v064 v065, a[216:219]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[212:215], v146 v147 v148 v149, v074 v075 v076 v077, a[212:215]
	v_mfma_f32_16x16x32_bf16 a[208:211], v146 v147 v148 v149, v078 v079 v080 v081, a[208:211]
	v_mfma_f32_16x16x32_bf16 a[204:207], v146 v147 v148 v149, v082 v083 v084 v085, a[204:207]
	ds_read_b128 v022 v023 v024 v025, v119 offset:34816	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[200:203], v146 v147 v148 v149, v086 v087 v088 v089, a[200:203]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[196:199], v146 v147 v148 v149, v090 v091 v092 v093, a[196:199]
	v_mfma_f32_16x16x32_bf16 a[192:195], v146 v147 v148 v149, v094 v095 v096 v097, a[192:195]
	v_mfma_f32_16x16x32_bf16 a[188:191], v002 v003 v004 v005, v070 v071 v072 v073, a[188:191]
	v_lshlrev_b32_e32 v006, 1, v006	;.loc	2 652 24
	s_mov_b32 m0, s22	;.loc	2 567 0
	buffer_load_dwordx4 v006, s[8:11], 0 offen sc0 lds
	ds_read_b128 v014 v015 v016 v017, v119 offset:35840	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[184:187], v002 v003 v004 v005, v062 v063 v064 v065, a[184:187]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[180:183], v002 v003 v004 v005, v074 v075 v076 v077, a[180:183]
	v_mfma_f32_16x16x32_bf16 a[176:179], v002 v003 v004 v005, v078 v079 v080 v081, a[176:179]
	v_mfma_f32_16x16x32_bf16 a[172:175], v002 v003 v004 v005, v082 v083 v084 v085, a[172:175]
	ds_read_b128 v010 v011 v012 v013, v119 offset:36864	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[168:171], v002 v003 v004 v005, v086 v087 v088 v089, a[168:171]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[164:167], v002 v003 v004 v005, v090 v091 v092 v093, a[164:167]
	v_mfma_f32_16x16x32_bf16 a[160:163], v002 v003 v004 v005, v094 v095 v096 v097, a[160:163]
	v_add_lshl_u32 v002, v110, s25, 14	;.loc	2 355 0
	v_add_lshl_u32 v018, v114, s25, 14
	v_or3_b32 v002, v002, v112, v105
	v_or3_b32 v018, v018, v116, v105
	v_lshlrev_b32_e32 v002, 1, v002	;.loc	2 652 24
	v_lshlrev_b32_e32 v018, 1, v018	;.loc	2 615 24
	v_mfma_f32_16x16x32_bf16 a[156:159], v150 v151 v152 v153, v070 v071 v072 v073, a[156:159]	;.loc	2 939 0
	s_mov_b32 m0, s21	;.loc	2 567 0
	buffer_load_dwordx4 v002, s[8:11], 0 offen sc0 lds
	ds_read_b128 v006 v007 v008 v009, v119 offset:37888	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[152:155], v150 v151 v152 v153, v062 v063 v064 v065, a[152:155]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[148:151], v150 v151 v152 v153, v074 v075 v076 v077, a[148:151]
	v_mfma_f32_16x16x32_bf16 a[144:147], v150 v151 v152 v153, v078 v079 v080 v081, a[144:147]
	v_mfma_f32_16x16x32_bf16 a[140:143], v150 v151 v152 v153, v082 v083 v084 v085, a[140:143]
	ds_read_b128 v002 v003 v004 v005, v119 offset:38912	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[136:139], v150 v151 v152 v153, v086 v087 v088 v089, a[136:139]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[132:135], v150 v151 v152 v153, v090 v091 v092 v093, a[132:135]
	v_mfma_f32_16x16x32_bf16 a[128:131], v150 v151 v152 v153, v094 v095 v096 v097, a[128:131]
	v_mfma_f32_16x16x32_bf16 a[124:127], v034 v035 v036 v037, v070 v071 v072 v073, a[124:127]
	s_mov_b32 m0, s19	;.loc	2 567 0
	buffer_load_dwordx4 v018, s[4:7], 0 offen sc0 lds
	ds_read_b128 v018 v019 v020 v021, v121 offset:32768	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[120:123], v034 v035 v036 v037, v062 v063 v064 v065, a[120:123]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[116:119], v034 v035 v036 v037, v074 v075 v076 v077, a[116:119]
	v_mfma_f32_16x16x32_bf16 a[112:115], v034 v035 v036 v037, v078 v079 v080 v081, a[112:115]
	v_mfma_f32_16x16x32_bf16 a[108:111], v034 v035 v036 v037, v082 v083 v084 v085, a[108:111]
	ds_read_b128 v026 v027 v028 v029, v121 offset:33792	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[104:107], v034 v035 v036 v037, v086 v087 v088 v089, a[104:107]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[100:103], v034 v035 v036 v037, v090 v091 v092 v093, a[100:103]
	v_mfma_f32_16x16x32_bf16 a[96:99], v034 v035 v036 v037, v094 v095 v096 v097, a[96:99]
	v_add_lshl_u32 v034, v118, s25, 14	;.loc	2 355 0
	v_or3_b32 v034, v034, v120, v105
	v_lshlrev_b32_e32 v034, 1, v034	;.loc	2 615 24
	v_mfma_f32_16x16x32_bf16 a[92:95], v050 v051 v052 v053, v070 v071 v072 v073, a[92:95]	;.loc	2 939 0
	s_mov_b32 m0, s15	;.loc	2 567 0
	buffer_load_dwordx4 v034, s[4:7], 0 offen sc0 lds
	ds_read_b128 v034 v035 v036 v037, v121 offset:34816	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[88:91], v050 v051 v052 v053, v062 v063 v064 v065, a[88:91]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[84:87], v050 v051 v052 v053, v074 v075 v076 v077, a[84:87]
	v_mfma_f32_16x16x32_bf16 a[80:83], v050 v051 v052 v053, v078 v079 v080 v081, a[80:83]
	v_mfma_f32_16x16x32_bf16 a[76:79], v050 v051 v052 v053, v082 v083 v084 v085, a[76:79]
	ds_read_b128 v042 v043 v044 v045, v121 offset:35840	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[72:75], v050 v051 v052 v053, v086 v087 v088 v089, a[72:75]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[68:71], v050 v051 v052 v053, v090 v091 v092 v093, a[68:71]
	v_mfma_f32_16x16x32_bf16 a[64:67], v050 v051 v052 v053, v094 v095 v096 v097, a[64:67]
	v_add_lshl_u32 v050, v122, s25, 14	;.loc	2 355 0
	v_or3_b32 v050, v050, v124, v105
	v_lshlrev_b32_e32 v050, 1, v050	;.loc	2 615 24
	v_mfma_f32_16x16x32_bf16 a[60:63], v058 v059 v060 v061, v070 v071 v072 v073, a[60:63]	;.loc	2 939 0
	s_mov_b32 m0, s14	;.loc	2 567 0
	buffer_load_dwordx4 v050, s[4:7], 0 offen sc0 lds
	ds_read_b128 v050 v051 v052 v053, v121 offset:36864	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[56:59], v058 v059 v060 v061, v062 v063 v064 v065, a[56:59]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[52:55], v058 v059 v060 v061, v074 v075 v076 v077, a[52:55]
	v_mfma_f32_16x16x32_bf16 a[48:51], v058 v059 v060 v061, v078 v079 v080 v081, a[48:51]
	v_mfma_f32_16x16x32_bf16 a[44:47], v058 v059 v060 v061, v082 v083 v084 v085, a[44:47]
	ds_read_b128 v054 v055 v056 v057, v121 offset:37888	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[40:43], v058 v059 v060 v061, v086 v087 v088 v089, a[40:43]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[36:39], v058 v059 v060 v061, v090 v091 v092 v093, a[36:39]
	v_mfma_f32_16x16x32_bf16 a[32:35], v058 v059 v060 v061, v094 v095 v096 v097, a[32:35]
	v_add_lshl_u32 v058, v126, s25, 14	;.loc	2 355 0
	v_or3_b32 v058, v058, v128, v105
	v_lshlrev_b32_e32 v058, 1, v058	;.loc	2 615 24
	v_mfma_f32_16x16x32_bf16 a[28:31], v066 v067 v068 v069, v070 v071 v072 v073, a[28:31]	;.loc	2 939 0
	s_mov_b32 m0, s3	;.loc	2 567 0
	buffer_load_dwordx4 v058, s[4:7], 0 offen sc0 lds
	ds_read_b128 v058 v059 v060 v061, v121 offset:38912	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[24:27], v066 v067 v068 v069, v062 v063 v064 v065, a[24:27]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[20:23], v066 v067 v068 v069, v074 v075 v076 v077, a[20:23]
	v_mfma_f32_16x16x32_bf16 a[16:19], v066 v067 v068 v069, v078 v079 v080 v081, a[16:19]
	v_mfma_f32_16x16x32_bf16 a[12:15], v066 v067 v068 v069, v082 v083 v084 v085, a[12:15]
	ds_read_b128 v062 v063 v064 v065, v121 offset:39936	;.loc	1 317 12
	v_lshlrev_b32_e32 v070, 5, v099	;.loc	2 719 0
	v_mfma_f32_16x16x32_bf16 a[8:11], v066 v067 v068 v069, v086 v087 v088 v089, a[8:11]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[4:7], v066 v067 v068 v069, v090 v091 v092 v093, a[4:7]
	v_mfma_f32_16x16x32_bf16 a[0:3], v066 v067 v068 v069, v094 v095 v096 v097, a[0:3]
	s_add_i32 s21, s2, 0x3f80
	s_add_i32 s22, s2, 0xc0	;.loc	2 1144 0
	v_add_u32_e32 v066, 0xa0, v113
	s_mov_b64 s[14:15], 1
	s_mov_b64 s[2:3], 0
	v_lshlrev_b32_e32 v067, 1, v115
	v_lshlrev_b32_e32 v068, 1, v070
	s_waitcnt vmcnt(8) lgkmcnt(0)	;.loc	2 421 0
	s_barrier
.LBB0_1:
	s_mov_b64 s[24:25], s[14:15]	;.loc	2 0 0 is_stmt 0
	s_add_i32 s15, s22, s2	;.loc	2 1156 31 is_stmt 1
	v_add_u32_e32 v069, s2, v066	;.loc	3 875 16
	s_min_i32 s23, s15, s21	;.loc	2 1156 31
	v_lshrrev_b32_e32 v070, 6, v069	;.loc	2 353 0
	s_lshl_b32 s15, s24, 15	;.loc	1 317 12
	s_xor_b32 s14, s24, 1	;.loc	2 1150 0
	v_and_b32_e32 v069, 56, v069	;.loc	2 354 0
	v_add_lshl_u32 v071, v070, v098, 14	;.loc	2 355 0
	v_or_b32_e32 v072, s15, v103	;.loc	1 317 12
	v_add_lshl_u32 v073, v070, v102, 14	;.loc	2 355 0
	v_add_lshl_u32 v074, v070, v106, 14
	v_add_lshl_u32 v075, v070, v110, 14
	v_add_lshl_u32 v076, v070, v114, 14
	v_add_lshl_u32 v077, v070, v118, 14
	v_add_lshl_u32 v078, v070, v122, 14
	v_add_lshl_u32 v070, v070, v126, 14
	s_lshr_b32 s30, s23, 6	;.loc	2 353 0
	s_lshl_b32 s24, s14, 15	;.loc	2 582 0
	v_bitop3_b32 v079, s23, 63, v105 bitop3:0xc8	;.loc	2 354 0
	v_or3_b32 v071, v071, v100, v069	;.loc	2 355 0
	v_add_u32_e32 v080, v072, v111	;.loc	1 317 12
	v_add_u32_e32 v113, v072, v067
	v_or3_b32 v073, v073, v104, v069	;.loc	2 355 0
	v_or3_b32 v074, v074, v108, v069
	v_or3_b32 v075, v075, v112, v069
	v_or3_b32 v076, v076, v116, v069
	v_add3_u32 v115, v072, v068, s20	;.loc	1 317 12
	v_or3_b32 v072, v077, v120, v069	;.loc	2 355 0
	v_or3_b32 v077, v078, v124, v069
	v_or3_b32 v069, v070, v128, v069
	v_add_lshl_u32 v070, s30, v098, 14
	v_add_lshl_u32 v081, s30, v102, 14
	v_add_lshl_u32 v082, s30, v106, 14
	v_add_lshl_u32 v083, s30, v110, 14
	v_add_lshl_u32 v084, s30, v114, 14
	v_add_lshl_u32 v085, s30, v118, 14
	v_add_lshl_u32 v086, s30, v122, 14
	v_add_lshl_u32 v087, s30, v126, 14
	s_add_i32 s31, s19, s24	;.loc	2 582 0
	v_or_b32_e32 v078, s24, v103	;.loc	1 317 12
	v_lshlrev_b32_e32 v071, 1, v071	;.loc	2 652 24
	v_lshlrev_b32_e32 v088, 1, v073
	v_lshlrev_b32_e32 v089, 1, v074
	v_lshlrev_b32_e32 v094, 1, v075
	v_or3_b32 v123, v070, v100, v079	;.loc	2 355 0
	v_or3_b32 v129, v081, v104, v079
	v_or3_b32 v166, v082, v108, v079
	v_or3_b32 v167, v083, v112, v079
	v_or3_b32 v168, v084, v116, v079
	v_or3_b32 v170, v085, v120, v079
	v_or3_b32 v171, v086, v124, v079
	v_or3_b32 v172, v087, v128, v079
	s_waitcnt lgkmcnt(7)	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[252:255], v046 v047 v048 v049, v018 v019 v020 v021, a[252:255]
	s_add_i32 s15, s15, s19	;.loc	2 582 0
	s_add_i32 s30, s31, 0x4000
	s_add_i32 s33, s31, 0x14000	;.loc	2 583 0
	s_add_i32 s34, s31, 0x5000	;.loc	2 582 0
	s_add_i32 s35, s31, 0x15000	;.loc	2 583 0
	s_add_i32 s36, s31, 0x6000	;.loc	2 582 0
	s_add_i32 s37, s31, 0x16000	;.loc	2 583 0
	s_add_i32 s38, s31, 0x7000	;.loc	2 582 0
	s_add_i32 s31, s31, 0x17000	;.loc	2 583 0
	v_lshlrev_b32_e32 v117, 1, v076	;.loc	2 615 24
	v_lshlrev_b32_e32 v119, 1, v072
	v_lshlrev_b32_e32 v121, 1, v077
	v_lshlrev_b32_e32 v069, 1, v069
	v_add_u32_e32 v125, v078, v111	;.loc	1 317 12
	v_add_u32_e32 v127, v078, v067
	v_add3_u32 v169, v078, v068, s20
	s_mov_b32 m0, s33	;.loc	2 567 0
	buffer_load_dwordx4 v071, s[8:11], 0 offen sc0 lds
	ds_read_b128 v070 v071 v072 v073, v080 offset:16384	;.loc	1 317 12
	s_waitcnt lgkmcnt(7)	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[248:251], v046 v047 v048 v049, v026 v027 v028 v029, a[248:251]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 a[244:247], v046 v047 v048 v049, v034 v035 v036 v037, a[244:247]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 a[240:243], v046 v047 v048 v049, v042 v043 v044 v045, a[240:243]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 a[236:239], v046 v047 v048 v049, v050 v051 v052 v053, a[236:239]
	ds_read_b128 v074 v075 v076 v077, v113 offset:16384	;.loc	1 317 12
	s_waitcnt lgkmcnt(4)	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[232:235], v046 v047 v048 v049, v054 v055 v056 v057, a[232:235]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 a[228:231], v046 v047 v048 v049, v058 v059 v060 v061, a[228:231]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 a[224:227], v046 v047 v048 v049, v062 v063 v064 v065, a[224:227]
	v_mfma_f32_16x16x32_bf16 a[220:223], v038 v039 v040 v041, v018 v019 v020 v021, a[220:223]
	s_mov_b32 m0, s35	;.loc	2 567 0
	buffer_load_dwordx4 v088, s[8:11], 0 offen sc0 lds
	ds_read_b128 v078 v079 v080 v081, v113 offset:17408	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[216:219], v038 v039 v040 v041, v026 v027 v028 v029, a[216:219]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[212:215], v038 v039 v040 v041, v034 v035 v036 v037, a[212:215]
	v_mfma_f32_16x16x32_bf16 a[208:211], v038 v039 v040 v041, v042 v043 v044 v045, a[208:211]
	v_mfma_f32_16x16x32_bf16 a[204:207], v038 v039 v040 v041, v050 v051 v052 v053, a[204:207]
	ds_read_b128 v082 v083 v084 v085, v113 offset:18432	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[200:203], v038 v039 v040 v041, v054 v055 v056 v057, a[200:203]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[196:199], v038 v039 v040 v041, v058 v059 v060 v061, a[196:199]
	v_mfma_f32_16x16x32_bf16 a[192:195], v038 v039 v040 v041, v062 v063 v064 v065, a[192:195]
	v_mfma_f32_16x16x32_bf16 a[188:191], v030 v031 v032 v033, v018 v019 v020 v021, a[188:191]
	s_mov_b32 m0, s37	;.loc	2 567 0
	buffer_load_dwordx4 v089, s[8:11], 0 offen sc0 lds
	ds_read_b128 v086 v087 v088 v089, v113 offset:19456	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[184:187], v030 v031 v032 v033, v026 v027 v028 v029, a[184:187]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[180:183], v030 v031 v032 v033, v034 v035 v036 v037, a[180:183]
	v_mfma_f32_16x16x32_bf16 a[176:179], v030 v031 v032 v033, v042 v043 v044 v045, a[176:179]
	v_mfma_f32_16x16x32_bf16 a[172:175], v030 v031 v032 v033, v050 v051 v052 v053, a[172:175]
	ds_read_b128 v090 v091 v092 v093, v113 offset:20480	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[168:171], v030 v031 v032 v033, v054 v055 v056 v057, a[168:171]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[164:167], v030 v031 v032 v033, v058 v059 v060 v061, a[164:167]
	v_mfma_f32_16x16x32_bf16 a[160:163], v030 v031 v032 v033, v062 v063 v064 v065, a[160:163]
	v_mfma_f32_16x16x32_bf16 a[156:159], v022 v023 v024 v025, v018 v019 v020 v021, a[156:159]
	s_mov_b32 m0, s31	;.loc	2 567 0
	buffer_load_dwordx4 v094, s[8:11], 0 offen sc0 lds
	ds_read_b128 v094 v095 v096 v097, v113 offset:21504	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[152:155], v022 v023 v024 v025, v026 v027 v028 v029, a[152:155]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[148:151], v022 v023 v024 v025, v034 v035 v036 v037, a[148:151]
	v_mfma_f32_16x16x32_bf16 a[144:147], v022 v023 v024 v025, v042 v043 v044 v045, a[144:147]
	v_mfma_f32_16x16x32_bf16 a[140:143], v022 v023 v024 v025, v050 v051 v052 v053, a[140:143]
	ds_read_b128 v130 v131 v132 v133, v113 offset:22528	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[136:139], v022 v023 v024 v025, v054 v055 v056 v057, a[136:139]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[132:135], v022 v023 v024 v025, v058 v059 v060 v061, a[132:135]
	v_mfma_f32_16x16x32_bf16 a[128:131], v022 v023 v024 v025, v062 v063 v064 v065, a[128:131]
	v_mfma_f32_16x16x32_bf16 a[124:127], v014 v015 v016 v017, v018 v019 v020 v021, a[124:127]
	s_mov_b32 m0, s30	;.loc	2 567 0
	buffer_load_dwordx4 v117, s[4:7], 0 offen sc0 lds
	ds_read_b128 v134 v135 v136 v137, v115 offset:16384	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[120:123], v014 v015 v016 v017, v026 v027 v028 v029, a[120:123]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[116:119], v014 v015 v016 v017, v034 v035 v036 v037, a[116:119]
	v_mfma_f32_16x16x32_bf16 a[112:115], v014 v015 v016 v017, v042 v043 v044 v045, a[112:115]
	v_mfma_f32_16x16x32_bf16 a[108:111], v014 v015 v016 v017, v050 v051 v052 v053, a[108:111]
	ds_read_b128 v138 v139 v140 v141, v115 offset:17408	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[104:107], v014 v015 v016 v017, v054 v055 v056 v057, a[104:107]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[100:103], v014 v015 v016 v017, v058 v059 v060 v061, a[100:103]
	v_mfma_f32_16x16x32_bf16 a[96:99], v014 v015 v016 v017, v062 v063 v064 v065, a[96:99]
	v_mfma_f32_16x16x32_bf16 a[92:95], v010 v011 v012 v013, v018 v019 v020 v021, a[92:95]
	s_mov_b32 m0, s34	;.loc	2 567 0
	buffer_load_dwordx4 v119, s[4:7], 0 offen sc0 lds
	ds_read_b128 v142 v143 v144 v145, v115 offset:18432	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[88:91], v010 v011 v012 v013, v026 v027 v028 v029, a[88:91]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[84:87], v010 v011 v012 v013, v034 v035 v036 v037, a[84:87]
	v_mfma_f32_16x16x32_bf16 a[80:83], v010 v011 v012 v013, v042 v043 v044 v045, a[80:83]
	v_mfma_f32_16x16x32_bf16 a[76:79], v010 v011 v012 v013, v050 v051 v052 v053, a[76:79]
	ds_read_b128 v146 v147 v148 v149, v115 offset:19456	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[72:75], v010 v011 v012 v013, v054 v055 v056 v057, a[72:75]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[68:71], v010 v011 v012 v013, v058 v059 v060 v061, a[68:71]
	v_mfma_f32_16x16x32_bf16 a[64:67], v010 v011 v012 v013, v062 v063 v064 v065, a[64:67]
	v_mfma_f32_16x16x32_bf16 a[60:63], v006 v007 v008 v009, v018 v019 v020 v021, a[60:63]
	s_mov_b32 m0, s36	;.loc	2 567 0
	buffer_load_dwordx4 v121, s[4:7], 0 offen sc0 lds
	ds_read_b128 v150 v151 v152 v153, v115 offset:20480	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[56:59], v006 v007 v008 v009, v026 v027 v028 v029, a[56:59]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[52:55], v006 v007 v008 v009, v034 v035 v036 v037, a[52:55]
	v_mfma_f32_16x16x32_bf16 a[48:51], v006 v007 v008 v009, v042 v043 v044 v045, a[48:51]
	v_mfma_f32_16x16x32_bf16 a[44:47], v006 v007 v008 v009, v050 v051 v052 v053, a[44:47]
	ds_read_b128 v154 v155 v156 v157, v115 offset:21504	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[40:43], v006 v007 v008 v009, v054 v055 v056 v057, a[40:43]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[36:39], v006 v007 v008 v009, v058 v059 v060 v061, a[36:39]
	v_mfma_f32_16x16x32_bf16 a[32:35], v006 v007 v008 v009, v062 v063 v064 v065, a[32:35]
	v_mfma_f32_16x16x32_bf16 a[28:31], v002 v003 v004 v005, v018 v019 v020 v021, a[28:31]
	s_mov_b32 m0, s38	;.loc	2 567 0
	buffer_load_dwordx4 v069, s[4:7], 0 offen sc0 lds
	ds_read_b128 v158 v159 v160 v161, v115 offset:22528	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[24:27], v002 v003 v004 v005, v026 v027 v028 v029, a[24:27]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[20:23], v002 v003 v004 v005, v034 v035 v036 v037, a[20:23]
	v_mfma_f32_16x16x32_bf16 a[16:19], v002 v003 v004 v005, v042 v043 v044 v045, a[16:19]
	v_mfma_f32_16x16x32_bf16 a[12:15], v002 v003 v004 v005, v050 v051 v052 v053, a[12:15]
	ds_read_b128 v162 v163 v164 v165, v115 offset:23552	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[8:11], v002 v003 v004 v005, v054 v055 v056 v057, a[8:11]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[4:7], v002 v003 v004 v005, v058 v059 v060 v061, a[4:7]
	v_mfma_f32_16x16x32_bf16 a[0:3], v002 v003 v004 v005, v062 v063 v064 v065, a[0:3]
	v_lshlrev_b32_e32 v002, 1, v123	;.loc	2 652 24
	v_lshlrev_b32_e32 v003, 1, v129
	v_lshlrev_b32_e32 v004, 1, v166
	v_lshlrev_b32_e32 v005, 1, v167
	v_lshlrev_b32_e32 v018, 1, v168	;.loc	2 615 24
	v_lshlrev_b32_e32 v034, 1, v170
	v_lshlrev_b32_e32 v050, 1, v171
	v_lshlrev_b32_e32 v058, 1, v172
	s_add_i32 s23, s15, 0x10000	;.loc	2 583 0
	s_add_i32 s24, s15, 0x11000
	s_add_i32 s25, s15, 0x12000
	s_add_i32 s26, s15, 0x13000
	s_add_i32 s27, s15, 0x1000
	s_add_i32 s28, s15, 0x2000
	s_add_i32 s29, s15, 0x3000
	s_waitcnt vmcnt(8) lgkmcnt(0)	;.loc	2 421 0
	s_barrier
	s_waitcnt lgkmcnt(7)	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[252:255], v070 v071 v072 v073, v134 v135 v136 v137, a[252:255]
	s_mov_b32 m0, s23	;.loc	2 567 0
	buffer_load_dwordx4 v002, s[8:11], 0 offen sc0 lds
	ds_read_b128 v046 v047 v048 v049, v125	;.loc	1 317 12
	s_waitcnt lgkmcnt(7)	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[248:251], v070 v071 v072 v073, v138 v139 v140 v141, a[248:251]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 a[244:247], v070 v071 v072 v073, v142 v143 v144 v145, a[244:247]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 a[240:243], v070 v071 v072 v073, v146 v147 v148 v149, a[240:243]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 a[236:239], v070 v071 v072 v073, v150 v151 v152 v153, a[236:239]
	ds_read_b128 v038 v039 v040 v041, v127	;.loc	1 317 12
	s_waitcnt lgkmcnt(4)	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[232:235], v070 v071 v072 v073, v154 v155 v156 v157, a[232:235]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 a[228:231], v070 v071 v072 v073, v158 v159 v160 v161, a[228:231]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 a[224:227], v070 v071 v072 v073, v162 v163 v164 v165, a[224:227]
	v_mfma_f32_16x16x32_bf16 a[220:223], v074 v075 v076 v077, v134 v135 v136 v137, a[220:223]
	s_mov_b32 m0, s24	;.loc	2 567 0
	buffer_load_dwordx4 v003, s[8:11], 0 offen sc0 lds
	ds_read_b128 v030 v031 v032 v033, v127 offset:1024	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[216:219], v074 v075 v076 v077, v138 v139 v140 v141, a[216:219]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[212:215], v074 v075 v076 v077, v142 v143 v144 v145, a[212:215]
	v_mfma_f32_16x16x32_bf16 a[208:211], v074 v075 v076 v077, v146 v147 v148 v149, a[208:211]
	v_mfma_f32_16x16x32_bf16 a[204:207], v074 v075 v076 v077, v150 v151 v152 v153, a[204:207]
	ds_read_b128 v022 v023 v024 v025, v127 offset:2048	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[200:203], v074 v075 v076 v077, v154 v155 v156 v157, a[200:203]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[196:199], v074 v075 v076 v077, v158 v159 v160 v161, a[196:199]
	v_mfma_f32_16x16x32_bf16 a[192:195], v074 v075 v076 v077, v162 v163 v164 v165, a[192:195]
	v_mfma_f32_16x16x32_bf16 a[188:191], v078 v079 v080 v081, v134 v135 v136 v137, a[188:191]
	s_mov_b32 m0, s25	;.loc	2 567 0
	buffer_load_dwordx4 v004, s[8:11], 0 offen sc0 lds
	ds_read_b128 v014 v015 v016 v017, v127 offset:3072	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[184:187], v078 v079 v080 v081, v138 v139 v140 v141, a[184:187]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[180:183], v078 v079 v080 v081, v142 v143 v144 v145, a[180:183]
	v_mfma_f32_16x16x32_bf16 a[176:179], v078 v079 v080 v081, v146 v147 v148 v149, a[176:179]
	v_mfma_f32_16x16x32_bf16 a[172:175], v078 v079 v080 v081, v150 v151 v152 v153, a[172:175]
	ds_read_b128 v010 v011 v012 v013, v127 offset:4096	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[168:171], v078 v079 v080 v081, v154 v155 v156 v157, a[168:171]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[164:167], v078 v079 v080 v081, v158 v159 v160 v161, a[164:167]
	v_mfma_f32_16x16x32_bf16 a[160:163], v078 v079 v080 v081, v162 v163 v164 v165, a[160:163]
	v_mfma_f32_16x16x32_bf16 a[156:159], v082 v083 v084 v085, v134 v135 v136 v137, a[156:159]
	s_mov_b32 m0, s26	;.loc	2 567 0
	buffer_load_dwordx4 v005, s[8:11], 0 offen sc0 lds
	ds_read_b128 v006 v007 v008 v009, v127 offset:5120	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[152:155], v082 v083 v084 v085, v138 v139 v140 v141, a[152:155]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[148:151], v082 v083 v084 v085, v142 v143 v144 v145, a[148:151]
	v_mfma_f32_16x16x32_bf16 a[144:147], v082 v083 v084 v085, v146 v147 v148 v149, a[144:147]
	v_mfma_f32_16x16x32_bf16 a[140:143], v082 v083 v084 v085, v150 v151 v152 v153, a[140:143]
	ds_read_b128 v002 v003 v004 v005, v127 offset:6144	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[136:139], v082 v083 v084 v085, v154 v155 v156 v157, a[136:139]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[132:135], v082 v083 v084 v085, v158 v159 v160 v161, a[132:135]
	v_mfma_f32_16x16x32_bf16 a[128:131], v082 v083 v084 v085, v162 v163 v164 v165, a[128:131]
	v_mfma_f32_16x16x32_bf16 a[124:127], v086 v087 v088 v089, v134 v135 v136 v137, a[124:127]
	s_mov_b32 m0, s15	;.loc	2 567 0
	buffer_load_dwordx4 v018, s[4:7], 0 offen sc0 lds
	ds_read_b128 v018 v019 v020 v021, v169	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[120:123], v086 v087 v088 v089, v138 v139 v140 v141, a[120:123]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[116:119], v086 v087 v088 v089, v142 v143 v144 v145, a[116:119]
	v_mfma_f32_16x16x32_bf16 a[112:115], v086 v087 v088 v089, v146 v147 v148 v149, a[112:115]
	v_mfma_f32_16x16x32_bf16 a[108:111], v086 v087 v088 v089, v150 v151 v152 v153, a[108:111]
	ds_read_b128 v026 v027 v028 v029, v169 offset:1024	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[104:107], v086 v087 v088 v089, v154 v155 v156 v157, a[104:107]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[100:103], v086 v087 v088 v089, v158 v159 v160 v161, a[100:103]
	v_mfma_f32_16x16x32_bf16 a[96:99], v086 v087 v088 v089, v162 v163 v164 v165, a[96:99]
	v_mfma_f32_16x16x32_bf16 a[92:95], v090 v091 v092 v093, v134 v135 v136 v137, a[92:95]
	s_mov_b32 m0, s27	;.loc	2 567 0
	buffer_load_dwordx4 v034, s[4:7], 0 offen sc0 lds
	ds_read_b128 v034 v035 v036 v037, v169 offset:2048	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[88:91], v090 v091 v092 v093, v138 v139 v140 v141, a[88:91]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[84:87], v090 v091 v092 v093, v142 v143 v144 v145, a[84:87]
	v_mfma_f32_16x16x32_bf16 a[80:83], v090 v091 v092 v093, v146 v147 v148 v149, a[80:83]
	v_mfma_f32_16x16x32_bf16 a[76:79], v090 v091 v092 v093, v150 v151 v152 v153, a[76:79]
	ds_read_b128 v042 v043 v044 v045, v169 offset:3072	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[72:75], v090 v091 v092 v093, v154 v155 v156 v157, a[72:75]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[68:71], v090 v091 v092 v093, v158 v159 v160 v161, a[68:71]
	v_mfma_f32_16x16x32_bf16 a[64:67], v090 v091 v092 v093, v162 v163 v164 v165, a[64:67]
	v_mfma_f32_16x16x32_bf16 a[60:63], v094 v095 v096 v097, v134 v135 v136 v137, a[60:63]
	s_mov_b32 m0, s28	;.loc	2 567 0
	buffer_load_dwordx4 v050, s[4:7], 0 offen sc0 lds
	ds_read_b128 v050 v051 v052 v053, v169 offset:4096	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[56:59], v094 v095 v096 v097, v138 v139 v140 v141, a[56:59]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[52:55], v094 v095 v096 v097, v142 v143 v144 v145, a[52:55]
	v_mfma_f32_16x16x32_bf16 a[48:51], v094 v095 v096 v097, v146 v147 v148 v149, a[48:51]
	v_mfma_f32_16x16x32_bf16 a[44:47], v094 v095 v096 v097, v150 v151 v152 v153, a[44:47]
	ds_read_b128 v054 v055 v056 v057, v169 offset:5120	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[40:43], v094 v095 v096 v097, v154 v155 v156 v157, a[40:43]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[36:39], v094 v095 v096 v097, v158 v159 v160 v161, a[36:39]
	v_mfma_f32_16x16x32_bf16 a[32:35], v094 v095 v096 v097, v162 v163 v164 v165, a[32:35]
	v_mfma_f32_16x16x32_bf16 a[28:31], v130 v131 v132 v133, v134 v135 v136 v137, a[28:31]
	s_mov_b32 m0, s29	;.loc	2 567 0
	buffer_load_dwordx4 v058, s[4:7], 0 offen sc0 lds
	ds_read_b128 v058 v059 v060 v061, v169 offset:6144	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[24:27], v130 v131 v132 v133, v138 v139 v140 v141, a[24:27]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[20:23], v130 v131 v132 v133, v142 v143 v144 v145, a[20:23]
	v_mfma_f32_16x16x32_bf16 a[16:19], v130 v131 v132 v133, v146 v147 v148 v149, a[16:19]
	v_mfma_f32_16x16x32_bf16 a[12:15], v130 v131 v132 v133, v150 v151 v152 v153, a[12:15]
	ds_read_b128 v062 v063 v064 v065, v169 offset:7168	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[8:11], v130 v131 v132 v133, v154 v155 v156 v157, a[8:11]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[4:7], v130 v131 v132 v133, v158 v159 v160 v161, a[4:7]
	v_mfma_f32_16x16x32_bf16 a[0:3], v130 v131 v132 v133, v162 v163 v164 v165, a[0:3]
	s_add_u32 s2, s2, 64	;.loc	2 1144 0
	s_addc_u32 s3, s3, 0
	s_cmp_lg_u64 s[2:3], 0x3f40
	s_waitcnt vmcnt(8) lgkmcnt(0)	;.loc	2 421 0
	s_barrier
.JUMP.LBB0_1:
	s_cbranch_scc1 .LBB0_1	;.loc	2 1144 0
	s_load_dwordx2 s[0:1], s[0:1], 0x0	;.loc	1 289 24
	s_lshl_b32 s4, s14, 14	;.loc	1 0 0 is_stmt 0
	s_lshl_b32 s4, s4, 1	;.loc	1 317 12 is_stmt 1
	v_add3_u32 v067, v111, s4, v103
	s_waitcnt lgkmcnt(0)	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[252:255], v046 v047 v048 v049, v018 v019 v020 v021, a[252:255]
	ds_read_b128 v068 v069 v070 v071, v067 offset:16384	;.loc	1 317 12
	v_add3_u32 v067, v109, s4, v103
	v_mfma_f32_16x16x32_bf16 a[248:251], v046 v047 v048 v049, v026 v027 v028 v029, a[248:251]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[244:247], v046 v047 v048 v049, v034 v035 v036 v037, a[244:247]
	v_mfma_f32_16x16x32_bf16 a[240:243], v046 v047 v048 v049, v042 v043 v044 v045, a[240:243]
	v_mfma_f32_16x16x32_bf16 a[236:239], v046 v047 v048 v049, v050 v051 v052 v053, a[236:239]
	ds_read_b128 v072 v073 v074 v075, v067 offset:16384	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[232:235], v046 v047 v048 v049, v054 v055 v056 v057, a[232:235]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[228:231], v046 v047 v048 v049, v058 v059 v060 v061, a[228:231]
	v_mfma_f32_16x16x32_bf16 a[224:227], v046 v047 v048 v049, v062 v063 v064 v065, a[224:227]
	v_mfma_f32_16x16x32_bf16 a[220:223], v038 v039 v040 v041, v018 v019 v020 v021, a[220:223]
	ds_read_b128 v076 v077 v078 v079, v067 offset:17408	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[216:219], v038 v039 v040 v041, v026 v027 v028 v029, a[216:219]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[212:215], v038 v039 v040 v041, v034 v035 v036 v037, a[212:215]
	v_mfma_f32_16x16x32_bf16 a[208:211], v038 v039 v040 v041, v042 v043 v044 v045, a[208:211]
	v_mfma_f32_16x16x32_bf16 a[204:207], v038 v039 v040 v041, v050 v051 v052 v053, a[204:207]
	ds_read_b128 v080 v081 v082 v083, v067 offset:18432	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[200:203], v038 v039 v040 v041, v054 v055 v056 v057, a[200:203]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[196:199], v038 v039 v040 v041, v058 v059 v060 v061, a[196:199]
	v_mfma_f32_16x16x32_bf16 a[192:195], v038 v039 v040 v041, v062 v063 v064 v065, a[192:195]
	v_mfma_f32_16x16x32_bf16 a[188:191], v030 v031 v032 v033, v018 v019 v020 v021, a[188:191]
	ds_read_b128 v084 v085 v086 v087, v067 offset:19456	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[184:187], v030 v031 v032 v033, v026 v027 v028 v029, a[184:187]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[180:183], v030 v031 v032 v033, v034 v035 v036 v037, a[180:183]
	v_mfma_f32_16x16x32_bf16 a[176:179], v030 v031 v032 v033, v042 v043 v044 v045, a[176:179]
	v_mfma_f32_16x16x32_bf16 a[172:175], v030 v031 v032 v033, v050 v051 v052 v053, a[172:175]
	ds_read_b128 v046 v047 v048 v049, v067 offset:20480	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[168:171], v030 v031 v032 v033, v054 v055 v056 v057, a[168:171]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[164:167], v030 v031 v032 v033, v058 v059 v060 v061, a[164:167]
	v_mfma_f32_16x16x32_bf16 a[160:163], v030 v031 v032 v033, v062 v063 v064 v065, a[160:163]
	v_mfma_f32_16x16x32_bf16 a[156:159], v022 v023 v024 v025, v018 v019 v020 v021, a[156:159]
	ds_read_b128 v038 v039 v040 v041, v067 offset:21504	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[152:155], v022 v023 v024 v025, v026 v027 v028 v029, a[152:155]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[148:151], v022 v023 v024 v025, v034 v035 v036 v037, a[148:151]
	v_mfma_f32_16x16x32_bf16 a[144:147], v022 v023 v024 v025, v042 v043 v044 v045, a[144:147]
	v_mfma_f32_16x16x32_bf16 a[140:143], v022 v023 v024 v025, v050 v051 v052 v053, a[140:143]
	ds_read_b128 v030 v031 v032 v033, v067 offset:22528	;.loc	1 317 12
	v_add3_u32 v067, v107, s4, v103
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, -1
	s_and_b32 s1, s1, 0xffff	;.loc	1 289 24
	v_mfma_f32_16x16x32_bf16 a[136:139], v022 v023 v024 v025, v054 v055 v056 v057, a[136:139]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[132:135], v022 v023 v024 v025, v058 v059 v060 v061, a[132:135]
	v_mfma_f32_16x16x32_bf16 a[128:131], v022 v023 v024 v025, v062 v063 v064 v065, a[128:131]
	v_mfma_f32_16x16x32_bf16 a[124:127], v014 v015 v016 v017, v018 v019 v020 v021, a[124:127]
	ds_read_b128 v022 v023 v024 v025, v067 offset:16384	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[120:123], v014 v015 v016 v017, v026 v027 v028 v029, a[120:123]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[116:119], v014 v015 v016 v017, v034 v035 v036 v037, a[116:119]
	v_mfma_f32_16x16x32_bf16 a[112:115], v014 v015 v016 v017, v042 v043 v044 v045, a[112:115]
	v_mfma_f32_16x16x32_bf16 a[108:111], v014 v015 v016 v017, v050 v051 v052 v053, a[108:111]
	ds_read_b128 v088 v089 v090 v091, v067 offset:17408	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[104:107], v014 v015 v016 v017, v054 v055 v056 v057, a[104:107]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[100:103], v014 v015 v016 v017, v058 v059 v060 v061, a[100:103]
	v_mfma_f32_16x16x32_bf16 a[96:99], v014 v015 v016 v017, v062 v063 v064 v065, a[96:99]
	v_mfma_f32_16x16x32_bf16 a[92:95], v010 v011 v012 v013, v018 v019 v020 v021, a[92:95]
	ds_read_b128 v014 v015 v016 v017, v067 offset:18432	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[88:91], v010 v011 v012 v013, v026 v027 v028 v029, a[88:91]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[84:87], v010 v011 v012 v013, v034 v035 v036 v037, a[84:87]
	v_mfma_f32_16x16x32_bf16 a[80:83], v010 v011 v012 v013, v042 v043 v044 v045, a[80:83]
	v_mfma_f32_16x16x32_bf16 a[76:79], v010 v011 v012 v013, v050 v051 v052 v053, a[76:79]
	ds_read_b128 v092 v093 v094 v095, v067 offset:19456	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[72:75], v010 v011 v012 v013, v054 v055 v056 v057, a[72:75]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[68:71], v010 v011 v012 v013, v058 v059 v060 v061, a[68:71]
	v_mfma_f32_16x16x32_bf16 a[64:67], v010 v011 v012 v013, v062 v063 v064 v065, a[64:67]
	v_mfma_f32_16x16x32_bf16 a[60:63], v006 v007 v008 v009, v018 v019 v020 v021, a[60:63]
	ds_read_b128 v010 v011 v012 v013, v067 offset:20480	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[56:59], v006 v007 v008 v009, v026 v027 v028 v029, a[56:59]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[52:55], v006 v007 v008 v009, v034 v035 v036 v037, a[52:55]
	v_mfma_f32_16x16x32_bf16 a[48:51], v006 v007 v008 v009, v042 v043 v044 v045, a[48:51]
	v_mfma_f32_16x16x32_bf16 a[44:47], v006 v007 v008 v009, v050 v051 v052 v053, a[44:47]
	ds_read_b128 v102 v103 v104 v105, v067 offset:21504	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[40:43], v006 v007 v008 v009, v054 v055 v056 v057, a[40:43]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[36:39], v006 v007 v008 v009, v058 v059 v060 v061, a[36:39]
	v_mfma_f32_16x16x32_bf16 a[32:35], v006 v007 v008 v009, v062 v063 v064 v065, a[32:35]
	v_mfma_f32_16x16x32_bf16 a[28:31], v002 v003 v004 v005, v018 v019 v020 v021, a[28:31]
	ds_read_b128 v006 v007 v008 v009, v067 offset:22528	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[24:27], v002 v003 v004 v005, v026 v027 v028 v029, a[24:27]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[20:23], v002 v003 v004 v005, v034 v035 v036 v037, a[20:23]
	v_mfma_f32_16x16x32_bf16 a[16:19], v002 v003 v004 v005, v042 v043 v044 v045, a[16:19]
	v_mfma_f32_16x16x32_bf16 a[12:15], v002 v003 v004 v005, v050 v051 v052 v053, a[12:15]
	ds_read_b128 v018 v019 v020 v021, v067 offset:23552	;.loc	1 317 12
	v_mfma_f32_16x16x32_bf16 a[8:11], v002 v003 v004 v005, v054 v055 v056 v057, a[8:11]	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[4:7], v002 v003 v004 v005, v058 v059 v060 v061, a[4:7]
	v_mfma_f32_16x16x32_bf16 a[0:3], v002 v003 v004 v005, v062 v063 v064 v065, a[0:3]
	v_or_b32_e32 v066, 0x70, v001	;.loc	2 695 0
	s_waitcnt vmcnt(8) lgkmcnt(0)	;.loc	2 421 0
	s_barrier
	s_waitcnt lgkmcnt(7)	;.loc	2 939 0
	v_mfma_f32_16x16x32_bf16 a[252:255], v068 v069 v070 v071, v022 v023 v024 v025, a[252:255]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 a[248:251], v068 v069 v070 v071, v088 v089 v090 v091, a[248:251]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 a[244:247], v068 v069 v070 v071, v014 v015 v016 v017, a[244:247]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 a[240:243], v068 v069 v070 v071, v092 v093 v094 v095, a[240:243]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 a[236:239], v068 v069 v070 v071, v010 v011 v012 v013, a[236:239]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 a[232:235], v068 v069 v070 v071, v102 v103 v104 v105, a[232:235]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 a[228:231], v068 v069 v070 v071, v006 v007 v008 v009, a[228:231]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 a[224:227], v068 v069 v070 v071, v018 v019 v020 v021, a[224:227]
	v_mfma_f32_16x16x32_bf16 a[220:223], v072 v073 v074 v075, v022 v023 v024 v025, a[220:223]
	v_mfma_f32_16x16x32_bf16 a[216:219], v072 v073 v074 v075, v088 v089 v090 v091, a[216:219]
	v_mfma_f32_16x16x32_bf16 a[212:215], v072 v073 v074 v075, v014 v015 v016 v017, a[212:215]
	v_mfma_f32_16x16x32_bf16 a[208:211], v072 v073 v074 v075, v092 v093 v094 v095, a[208:211]
	v_mfma_f32_16x16x32_bf16 a[204:207], v072 v073 v074 v075, v010 v011 v012 v013, a[204:207]
	v_mfma_f32_16x16x32_bf16 a[200:203], v072 v073 v074 v075, v102 v103 v104 v105, a[200:203]
	v_mfma_f32_16x16x32_bf16 a[196:199], v072 v073 v074 v075, v006 v007 v008 v009, a[196:199]
	v_mfma_f32_16x16x32_bf16 a[192:195], v072 v073 v074 v075, v018 v019 v020 v021, a[192:195]
	v_mfma_f32_16x16x32_bf16 a[188:191], v076 v077 v078 v079, v022 v023 v024 v025, a[188:191]
	v_mfma_f32_16x16x32_bf16 a[184:187], v076 v077 v078 v079, v088 v089 v090 v091, a[184:187]
	v_mfma_f32_16x16x32_bf16 a[180:183], v076 v077 v078 v079, v014 v015 v016 v017, a[180:183]
	v_mfma_f32_16x16x32_bf16 a[176:179], v076 v077 v078 v079, v092 v093 v094 v095, a[176:179]
	v_mfma_f32_16x16x32_bf16 a[172:175], v076 v077 v078 v079, v010 v011 v012 v013, a[172:175]
	v_mfma_f32_16x16x32_bf16 a[168:171], v076 v077 v078 v079, v102 v103 v104 v105, a[168:171]
	v_mfma_f32_16x16x32_bf16 a[164:167], v076 v077 v078 v079, v006 v007 v008 v009, a[164:167]
	v_mfma_f32_16x16x32_bf16 a[160:163], v076 v077 v078 v079, v018 v019 v020 v021, a[160:163]
	v_mfma_f32_16x16x32_bf16 a[156:159], v080 v081 v082 v083, v022 v023 v024 v025, a[156:159]
	v_mfma_f32_16x16x32_bf16 a[152:155], v080 v081 v082 v083, v088 v089 v090 v091, a[152:155]
	v_mfma_f32_16x16x32_bf16 a[148:151], v080 v081 v082 v083, v014 v015 v016 v017, a[148:151]
	v_mfma_f32_16x16x32_bf16 a[144:147], v080 v081 v082 v083, v092 v093 v094 v095, a[144:147]
	v_mfma_f32_16x16x32_bf16 a[140:143], v080 v081 v082 v083, v010 v011 v012 v013, a[140:143]
	v_mfma_f32_16x16x32_bf16 a[136:139], v080 v081 v082 v083, v102 v103 v104 v105, a[136:139]
	v_mfma_f32_16x16x32_bf16 a[132:135], v080 v081 v082 v083, v006 v007 v008 v009, a[132:135]
	v_mfma_f32_16x16x32_bf16 a[128:131], v080 v081 v082 v083, v018 v019 v020 v021, a[128:131]
	v_mfma_f32_16x16x32_bf16 a[124:127], v084 v085 v086 v087, v022 v023 v024 v025, a[124:127]
	v_mfma_f32_16x16x32_bf16 a[120:123], v084 v085 v086 v087, v088 v089 v090 v091, a[120:123]
	v_mfma_f32_16x16x32_bf16 a[116:119], v084 v085 v086 v087, v014 v015 v016 v017, a[116:119]
	v_mfma_f32_16x16x32_bf16 a[112:115], v084 v085 v086 v087, v092 v093 v094 v095, a[112:115]
	v_mfma_f32_16x16x32_bf16 a[108:111], v084 v085 v086 v087, v010 v011 v012 v013, a[108:111]
	v_mfma_f32_16x16x32_bf16 a[104:107], v084 v085 v086 v087, v102 v103 v104 v105, a[104:107]
	v_mfma_f32_16x16x32_bf16 a[100:103], v084 v085 v086 v087, v006 v007 v008 v009, a[100:103]
	v_mfma_f32_16x16x32_bf16 a[96:99], v084 v085 v086 v087, v018 v019 v020 v021, a[96:99]
	v_mfma_f32_16x16x32_bf16 a[92:95], v046 v047 v048 v049, v022 v023 v024 v025, a[92:95]
	v_mfma_f32_16x16x32_bf16 a[88:91], v046 v047 v048 v049, v088 v089 v090 v091, a[88:91]
	v_mfma_f32_16x16x32_bf16 a[84:87], v046 v047 v048 v049, v014 v015 v016 v017, a[84:87]
	v_mfma_f32_16x16x32_bf16 a[80:83], v046 v047 v048 v049, v092 v093 v094 v095, a[80:83]
	v_mfma_f32_16x16x32_bf16 a[76:79], v046 v047 v048 v049, v010 v011 v012 v013, a[76:79]
	v_mfma_f32_16x16x32_bf16 a[72:75], v046 v047 v048 v049, v102 v103 v104 v105, a[72:75]
	v_mfma_f32_16x16x32_bf16 a[68:71], v046 v047 v048 v049, v006 v007 v008 v009, a[68:71]
	v_mfma_f32_16x16x32_bf16 a[64:67], v046 v047 v048 v049, v018 v019 v020 v021, a[64:67]
	v_mfma_f32_16x16x32_bf16 a[60:63], v038 v039 v040 v041, v022 v023 v024 v025, a[60:63]
	v_mfma_f32_16x16x32_bf16 a[56:59], v038 v039 v040 v041, v088 v089 v090 v091, a[56:59]
	v_mfma_f32_16x16x32_bf16 a[52:55], v038 v039 v040 v041, v014 v015 v016 v017, a[52:55]
	v_mfma_f32_16x16x32_bf16 a[48:51], v038 v039 v040 v041, v092 v093 v094 v095, a[48:51]
	v_mfma_f32_16x16x32_bf16 a[44:47], v038 v039 v040 v041, v010 v011 v012 v013, a[44:47]
	v_mfma_f32_16x16x32_bf16 a[40:43], v038 v039 v040 v041, v102 v103 v104 v105, a[40:43]
	v_mfma_f32_16x16x32_bf16 a[36:39], v038 v039 v040 v041, v006 v007 v008 v009, a[36:39]
	v_mfma_f32_16x16x32_bf16 a[32:35], v038 v039 v040 v041, v018 v019 v020 v021, a[32:35]
	v_mfma_f32_16x16x32_bf16 a[28:31], v030 v031 v032 v033, v022 v023 v024 v025, a[28:31]
	v_mfma_f32_16x16x32_bf16 a[24:27], v030 v031 v032 v033, v088 v089 v090 v091, a[24:27]
	v_mfma_f32_16x16x32_bf16 a[20:23], v030 v031 v032 v033, v014 v015 v016 v017, a[20:23]
	v_mfma_f32_16x16x32_bf16 a[16:19], v030 v031 v032 v033, v092 v093 v094 v095, a[16:19]
	v_mfma_f32_16x16x32_bf16 a[12:15], v030 v031 v032 v033, v010 v011 v012 v013, a[12:15]
	v_mfma_f32_16x16x32_bf16 a[8:11], v030 v031 v032 v033, v102 v103 v104 v105, a[8:11]
	v_mfma_f32_16x16x32_bf16 a[4:7], v030 v031 v032 v033, v006 v007 v008 v009, a[4:7]
	v_mfma_f32_16x16x32_bf16 a[0:3], v030 v031 v032 v033, v018 v019 v020 v021, a[0:3]
	v_lshlrev_b32_e32 v002, 2, v101	;.loc	2 1298 0
	v_accvgpr_read_b32 v003, a252	;.loc	2 1308 22
	v_and_or_b32 v004, v002, 12, v001	;.loc	2 1306 0
	v_cvt_pk_bf16_f32 v005, v003, s0	;.loc	2 1309 0
	v_lshlrev_b32_e32 v003, 1, v099	;.loc	1 330 12
	v_lshl_or_b32 v004, v004, 9, v003
	s_barrier	;.loc	2 1300 0
	ds_write_b16 v004, v005	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a253	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:512	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a254	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:1024	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a255	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:1536	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a248	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:32	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a249	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:544	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a250	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:1056	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a251	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:1568	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a244	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:64	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a245	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:576	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a246	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:1088	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a247	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:1600	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a240	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:96	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a241	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:608	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a242	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:1120	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a243	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:1632	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a236	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:128	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a237	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:640	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a238	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:1152	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a239	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:1664	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a232	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:160	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a233	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:672	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a234	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:1184	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a235	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:1696	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a228	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:192	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a229	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:704	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a230	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:1216	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a231	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:1728	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a224	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:224	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a225	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:736	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a226	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:1248	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a227	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:1760	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a220	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:8192	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a221	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:8704	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a222	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:9216	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a223	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:9728	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a216	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:8224	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a217	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:8736	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a218	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:9248	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a219	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:9760	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a212	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:8256	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a213	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:8768	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a214	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:9280	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a215	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:9792	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a208	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:8288	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a209	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:8800	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a210	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:9312	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a211	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:9824	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a204	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:8320	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a205	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:8832	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a206	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:9344	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a207	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:9856	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a200	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:8352	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a201	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:8864	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a202	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:9376	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a203	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:9888	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a196	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:8384	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a197	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:8896	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a198	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:9408	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a199	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:9920	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a192	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:8416	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a193	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:8928	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a194	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:9440	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a195	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:9952	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a188	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:16384	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a189	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:16896	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a190	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:17408	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a191	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:17920	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a184	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:16416	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a185	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:16928	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a186	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:17440	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a187	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:17952	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a180	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:16448	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a181	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:16960	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a182	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:17472	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a183	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:17984	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a176	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:16480	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a177	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:16992	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a178	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:17504	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a179	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:18016	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a172	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:16512	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a173	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:17024	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a174	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:17536	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a175	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:18048	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a168	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:16544	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a169	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:17056	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a170	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:17568	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a171	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:18080	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a164	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:16576	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a165	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:17088	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a166	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:17600	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a167	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:18112	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a160	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:16608	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a161	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:17120	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a162	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:17632	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a163	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v004, v005 offset:18144	;.loc	1 330 12
	v_or3_b32 v001, v001, 48, v002	;.loc	2 1306 0
	v_accvgpr_read_b32 v005, a156	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	v_lshl_or_b32 v001, v001, 9, v003	;.loc	1 330 12
	ds_write_b16 v001, v005
	v_accvgpr_read_b32 v005, a157	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:512	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a158	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:1024	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a159	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:1536	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a152	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:32	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a153	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:544	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a154	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:1056	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a155	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:1568	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a148	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:64	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a149	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:576	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a150	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:1088	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a151	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:1600	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a144	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:96	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a145	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:608	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a146	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:1120	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a147	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:1632	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a140	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:128	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a141	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:640	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a142	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:1152	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a143	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:1664	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a136	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:160	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a137	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:672	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a138	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:1184	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a139	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:1696	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a132	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:192	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a133	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:704	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a134	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:1216	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a135	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:1728	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a128	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:224	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a129	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:736	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a130	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:1248	;.loc	1 330 12
	v_accvgpr_read_b32 v005, a131	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v005, v005, s0	;.loc	2 1309 0
	ds_write_b16 v001, v005 offset:1760	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a124	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:32768	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a125	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:33280	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a126	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:33792	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a127	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:34304	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a120	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:32800	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a121	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:33312	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a122	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:33824	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a123	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:34336	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a116	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:32832	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a117	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:33344	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a118	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:33856	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a119	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:34368	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a112	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:32864	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a113	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:33376	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a114	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:33888	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a115	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:34400	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a108	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:32896	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a109	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:33408	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a110	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:33920	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a111	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:34432	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a104	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:32928	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a105	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:33440	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a106	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:33952	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a107	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:34464	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a100	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:32960	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a101	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:33472	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a102	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:33984	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a103	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:34496	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a96	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:32992	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a97	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:33504	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a98	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:34016	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a99	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:34528	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a92	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:40960	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a93	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:41472	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a94	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:41984	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a95	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:42496	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a88	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:40992	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a89	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:41504	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a90	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:42016	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a91	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:42528	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a84	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:41024	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a85	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:41536	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a86	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:42048	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a87	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:42560	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a80	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:41056	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a81	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:41568	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a82	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:42080	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a83	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:42592	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a76	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:41088	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a77	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:41600	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a78	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:42112	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a79	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:42624	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a72	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:41120	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a73	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:41632	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a74	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:42144	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a75	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:42656	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a68	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:41152	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a69	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:41664	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a70	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:42176	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a71	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:42688	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a64	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:41184	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a65	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:41696	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a66	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:42208	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a67	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:42720	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a60	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:49152	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a61	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:49664	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a62	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:50176	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a63	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:50688	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a56	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:49184	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a57	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:49696	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a58	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:50208	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a59	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:50720	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a52	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:49216	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a53	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:49728	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a54	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:50240	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a55	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:50752	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a48	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:49248	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a49	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:49760	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a50	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:50272	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a51	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:50784	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a44	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:49280	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a45	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:49792	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a46	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:50304	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a47	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:50816	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a40	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:49312	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a41	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:49824	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a42	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:50336	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a43	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:50848	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a36	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:49344	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a37	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:49856	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a38	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:50368	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a39	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:50880	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a32	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:49376	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a33	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:49888	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a34	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:50400	;.loc	1 330 12
	v_accvgpr_read_b32 v001, a35	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v001, v001, s0	;.loc	2 1309 0
	ds_write_b16 v004, v001 offset:50912	;.loc	1 330 12
	v_or_b32_e32 v001, v002, v066	;.loc	2 1306 0
	v_accvgpr_read_b32 v002, a28	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	v_lshl_or_b32 v001, v001, 9, v003	;.loc	1 330 12
	ds_write_b16 v001, v002
	v_accvgpr_read_b32 v002, a29	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:512	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a30	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:1024	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a31	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:1536	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a24	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:32	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a25	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:544	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a26	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:1056	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a27	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:1568	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a20	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:64	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a21	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:576	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a22	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:1088	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a23	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:1600	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a16	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:96	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a17	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:608	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a18	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:1120	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a19	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:1632	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a12	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:128	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a13	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:640	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a14	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:1152	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a15	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:1664	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a8	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:160	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a9	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:672	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a10	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:1184	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a11	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:1696	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a4	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:192	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a5	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:704	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a6	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:1216	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a7	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:1728	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a0	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:224	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a1	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:736	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a2	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	ds_write_b16 v001, v002 offset:1248	;.loc	1 330 12
	v_accvgpr_read_b32 v002, a3	;.loc	2 1308 22
	v_cvt_pk_bf16_f32 v002, v002, s0	;.loc	2 1309 0
	v_lshrrev_b32_e32 v004, 5, v000	;.loc	2 1352 0
	ds_write_b16 v001, v002 offset:1760	;.loc	1 330 12
	v_or_b32_e32 v002, s18, v004	;.loc	2 1354 0
	v_mov_b32_e32 v003, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v002 v003	;.loc	2 1355 28
	v_lshlrev_b32_e32 v003, 3, v000
	s_waitcnt lgkmcnt(0)	;.loc	2 1349 0
	s_barrier
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_4:
	s_cbranch_execz .LBB0_4
	v_and_b32_e32 v000, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v001, 1, v000	;.loc	1 317 12
	v_lshl_or_b32 v001, v004, 9, v001
	ds_read_b128 v006 v007 v008 v009, v001
	v_or_b32_e32 v000, s16, v000	;.loc	2 1364 0
	v_lshlrev_b32_e32 v001, 13, v002
	v_add_lshl_u32 v000, v000, v001, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_4:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 8, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_6:
	s_cbranch_execz .LBB0_6
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_6:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 16, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_8:
	s_cbranch_execz .LBB0_8
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_8:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 24, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_10:
	s_cbranch_execz .LBB0_10
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_10:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 32, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_12:
	s_cbranch_execz .LBB0_12
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_12:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 40, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_14:
	s_cbranch_execz .LBB0_14
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_14:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 48, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_16:
	s_cbranch_execz .LBB0_16
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_16:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 56, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_18:
	s_cbranch_execz .LBB0_18
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_18:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 64, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_20:
	s_cbranch_execz .LBB0_20
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_20:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x48, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_22:
	s_cbranch_execz .LBB0_22
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_22:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x50, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_24:
	s_cbranch_execz .LBB0_24
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_24:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x58, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_26:
	s_cbranch_execz .LBB0_26
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_26:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x60, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_28:
	s_cbranch_execz .LBB0_28
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_28:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x68, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_30:
	s_cbranch_execz .LBB0_30
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_30:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x70, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_32:
	s_cbranch_execz .LBB0_32
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_32:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x78, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_34:
	s_cbranch_execz .LBB0_34
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_34:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x80, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_36:
	s_cbranch_execz .LBB0_36
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_36:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x88, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_38:
	s_cbranch_execz .LBB0_38
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_38:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x90, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_40:
	s_cbranch_execz .LBB0_40
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_40:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0x98, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_42:
	s_cbranch_execz .LBB0_42
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_42:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xa0, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_44:
	s_cbranch_execz .LBB0_44
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_44:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xa8, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_46:
	s_cbranch_execz .LBB0_46
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_46:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xb0, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_48:
	s_cbranch_execz .LBB0_48
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_48:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xb8, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_50:
	s_cbranch_execz .LBB0_50
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_50:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xc0, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_52:
	s_cbranch_execz .LBB0_52
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_52:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xc8, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_54:
	s_cbranch_execz .LBB0_54
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_54:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xd0, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_56:
	s_cbranch_execz .LBB0_56
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_56:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xd8, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_58:
	s_cbranch_execz .LBB0_58
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_58:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xe0, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_60:
	s_cbranch_execz .LBB0_60
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_60:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xe8, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_62:
	s_cbranch_execz .LBB0_62
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_62:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xf0, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_64:
	s_cbranch_execz .LBB0_64
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v005, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_64:
	s_or_b64 exec, exec, s[4:5]	;.loc	1 0 8 is_stmt 0
	v_or_b32_e32 v002, 0xf8, v004	;.loc	2 1352 0 is_stmt 1
	v_or_b32_e32 v000, s18, v002	;.loc	2 1354 0
	v_mov_b32_e32 v001, s17
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001	;.loc	2 1355 28
	s_and_saveexec_b64 s[4:5], vcc	;.loc	2 1356 0
.JUMP.LBB0_66:
	s_cbranch_execz .LBB0_66
	v_and_b32_e32 v001, 0xf8, v003	;.loc	2 1353 0
	v_lshlrev_b32_e32 v003, 1, v001	;.loc	1 317 12
	v_lshl_or_b32 v002, v002, 9, v003
	ds_read_b128 v002 v003 v004 v005, v002
	v_or_b32_e32 v001, s16, v001	;.loc	2 1364 0
	v_lshlrev_b32_e32 v000, 13, v000
	v_add_lshl_u32 v000, v001, v000, 1	;.loc	1 299 8
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v002 v003 v004 v005, v000, s[0:3], 0 offen
.LBB0_66:
	s_endpgm	;.loc	2 325 0
.Lfunc_end0:
	.size	hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0, .Lfunc_end0-hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0
	.cfi_endproc
	.set hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.num_vgpr, 173
	.set hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.num_agpr, 256
	.set hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.numbered_sgpr, 39
	.set hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.num_named_barrier, 0
	.set hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.private_seg_size, 0
	.set hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.uses_vcc, 1
	.set hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.uses_flat_scratch, 0
	.set hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.has_dyn_sized_stack, 0
	.set hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.has_recursion, 0
	.set hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.has_indirect_call, 0
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	.debug_abbrev,"",@progbits
	.byte	1
	.byte	17
	.byte	0
	.byte	37
	.byte	14
	.byte	19
	.byte	5
	.byte	3
	.byte	14
	.byte	16
	.byte	23
	.byte	17
	.byte	1
	.byte	18
	.byte	6
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_info,"",@progbits
...
	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
