	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0
	.p2align	8
	.type	hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0,@function
hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0:
	s_load_dword s12, s[0:1], 0x60
	s_load_dwordx2 s[4:5], s[0:1], 0x18
	s_load_dwordx2 s[8:9], s[0:1], 0x30
	s_ashr_i32 s14, s2, 31
	s_lshr_b32 s14, s14, 27
	s_add_i32 s14, s2, s14
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s13, s12, 31
	s_and_b32 s5, s5, 0xffff
	s_and_b32 s9, s9, 0xffff
	s_ashr_i32 s18, s14, 5
	s_and_b32 s19, s14, 0xffffffe0
	s_cmp_lg_u32 s2, s19
	s_cselect_b64 s[14:15], -1, 0
	s_cmp_lt_i32 s2, 0
	s_cselect_b64 s[16:17], -1, 0
	s_and_b64 s[14:15], s[16:17], s[14:15]
	v_lshrrev_b32_e32 v2, 6, v0
	s_subb_u32 s14, s18, 0
	s_lshl_b32 s20, s14, 8
	v_readfirstlane_b32 s14, v2
	v_lshrrev_b32_e32 v109, 2, v0
	s_sub_i32 s15, s2, s19
	s_lshl_b32 s21, s14, 10
	v_xor_b32_e32 v2, v109, v0
	s_lshl_b32 s2, s3, 14
	s_ashr_i32 s19, s20, 31
	s_lshl_b32 s18, s15, 8
	v_lshlrev_b32_e32 v2, 3, v2
	s_add_i32 s25, s21, 0x10000
	v_and_b32_e32 v97, 24, v2
	v_or_b32_e32 v2, s18, v109
	s_cmpk_lt_u32 s18, 0x2000
	v_lshlrev_b32_e32 v2, 14, v2
	s_cselect_b64 vcc, -1, 0
	v_or_b32_e32 v111, s2, v97
	v_cndmask_b32_e32 v98, 0, v2, vcc
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, -1
	v_add_lshl_u32 v2, v111, v98, 1
	v_or_b32_e32 v4, 64, v109
	s_mov_b32 s10, s6
	s_mov_b32 s11, s7
	;;#ASMSTART
	s_mov_b32 m0, s25
	buffer_load_dwordx4 v2, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	v_or_b32_e32 v2, s18, v4
	v_lshlrev_b32_e32 v2, 14, v2
	v_cndmask_b32_e32 v100, 0, v2, vcc
	v_add_lshl_u32 v2, v111, v100, 1
	v_or_b32_e32 v5, 0x80, v109
	s_add_i32 s24, s21, 0x11000
	;;#ASMSTART
	s_mov_b32 m0, s24
	buffer_load_dwordx4 v2, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	v_or_b32_e32 v2, s18, v5
	v_lshlrev_b32_e32 v2, 14, v2
	v_cndmask_b32_e32 v101, 0, v2, vcc
	v_add_lshl_u32 v2, v111, v101, 1
	v_or_b32_e32 v6, 0xc0, v109
	s_add_i32 s23, s21, 0x12000
	;;#ASMSTART
	s_mov_b32 m0, s23
	buffer_load_dwordx4 v2, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	v_or_b32_e32 v2, s18, v6
	v_lshlrev_b32_e32 v2, 14, v2
	v_cndmask_b32_e32 v102, 0, v2, vcc
	v_add_lshl_u32 v2, v111, v102, 1
	v_or_b32_e32 v7, 32, v111
	s_add_i32 s17, s21, 0x13000
	;;#ASMSTART
	s_mov_b32 m0, s17
	buffer_load_dwordx4 v2, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	v_add_lshl_u32 v2, v7, v98, 1
	s_add_i32 s14, s21, 0x14000
	;;#ASMSTART
	s_mov_b32 m0, s14
	buffer_load_dwordx4 v2, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	v_add_lshl_u32 v2, v7, v100, 1
	s_add_i32 s14, s21, 0x15000
	;;#ASMSTART
	s_mov_b32 m0, s14
	buffer_load_dwordx4 v2, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	v_add_lshl_u32 v2, v7, v101, 1
	s_add_i32 s14, s21, 0x16000
	;;#ASMSTART
	s_mov_b32 m0, s14
	buffer_load_dwordx4 v2, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	v_add_lshl_u32 v2, v7, v102, 1
	s_add_i32 s14, s21, 0x17000
	;;#ASMSTART
	s_mov_b32 m0, s14
	buffer_load_dwordx4 v2, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	v_or_b32_e32 v2, s20, v109
	v_mov_b32_e32 v3, s19
	v_lshlrev_b32_e32 v8, 14, v2
	v_cmp_gt_u64_e32 vcc, s[12:13], v[2:3]
	s_add_i32 s16, s21, 0x1000
	s_add_i32 s15, s21, 0x2000
	v_cndmask_b32_e32 v103, 0, v8, vcc
	v_add_lshl_u32 v2, v103, v111, 1
	;;#ASMSTART
	s_mov_b32 m0, s21
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	v_or_b32_e32 v2, s20, v4
	v_lshlrev_b32_e32 v4, 14, v2
	v_cmp_gt_u64_e32 vcc, s[12:13], v[2:3]
	s_add_i32 s14, s21, 0x3000
	s_add_i32 s27, s21, 0x4000
	v_cndmask_b32_e32 v104, 0, v4, vcc
	v_add_lshl_u32 v2, v104, v111, 1
	;;#ASMSTART
	s_mov_b32 m0, s16
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	v_or_b32_e32 v2, s20, v5
	v_lshlrev_b32_e32 v4, 14, v2
	v_cmp_gt_u64_e32 vcc, s[12:13], v[2:3]
	v_and_b32_e32 v1, 0x80, v0
	v_lshlrev_b32_e32 v34, 1, v0
	v_cndmask_b32_e32 v105, 0, v4, vcc
	v_add_lshl_u32 v2, v105, v111, 1
	;;#ASMSTART
	s_mov_b32 m0, s15
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	v_or_b32_e32 v2, s20, v6
	v_lshlrev_b32_e32 v4, 14, v2
	v_cmp_gt_u64_e32 vcc, s[12:13], v[2:3]
	v_and_b32_e32 v35, 15, v0
	v_lshrrev_b32_e32 v94, 4, v0
	v_cndmask_b32_e32 v106, 0, v4, vcc
	v_add_lshl_u32 v2, v106, v111, 1
	;;#ASMSTART
	s_mov_b32 m0, s14
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	v_add_lshl_u32 v2, v103, v7, 1
	;;#ASMSTART
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	v_add_lshl_u32 v2, v104, v7, 1
	s_add_i32 s27, s21, 0x5000
	;;#ASMSTART
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	v_add_lshl_u32 v2, v105, v7, 1
	s_add_i32 s27, s21, 0x6000
	;;#ASMSTART
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	v_add_lshl_u32 v2, v106, v7, 1
	s_add_i32 s27, s21, 0x7000
	;;#ASMSTART
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	v_or_b32_e32 v2, 64, v111
	s_add_i32 s27, s21, 0x18000
	v_add_lshl_u32 v3, v2, v98, 1
	;;#ASMSTART
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v3, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	s_add_i32 s27, s21, 0x19000
	v_add_lshl_u32 v3, v2, v100, 1
	;;#ASMSTART
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v3, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	s_add_i32 s27, s21, 0x1a000
	v_add_lshl_u32 v3, v2, v101, 1
	;;#ASMSTART
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v3, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	s_add_i32 s27, s21, 0x1b000
	v_add_lshl_u32 v3, v2, v102, 1
	;;#ASMSTART
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v3, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	s_add_i32 s27, s21, 0x8000
	v_add_lshl_u32 v3, v103, v2, 1
	;;#ASMSTART
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v3, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	s_add_i32 s27, s21, 0x9000
	v_add_lshl_u32 v3, v104, v2, 1
	s_mov_b32 s22, 0x10000
	;;#ASMSTART
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v3, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	s_add_i32 s27, s21, 0xa000
	v_add_lshl_u32 v3, v105, v2, 1
	v_add_lshl_u32 v2, v106, v2, 1
	s_movk_i32 s26, 0x80
	;;#ASMSTART
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v3, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	s_add_i32 s27, s21, 0xb000
	;;#ASMSTART
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v2, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	v_xor_b32_e32 v2, v94, v0
	v_lshlrev_b32_e32 v2, 4, v2
	v_and_b32_e32 v96, 48, v2
	v_lshlrev_b32_e32 v2, 5, v0
	v_and_b32_e32 v2, 0x11e0, v2
	v_lshlrev_b32_e32 v108, 1, v2
	v_or3_b32 v6, v1, v35, 16
	v_and_or_b32 v95, v34, s26, v35
	v_mov_b32_e32 v34, 0x10000
	v_or_b32_e32 v144, v108, v96
	v_lshlrev_b32_e32 v107, 6, v6
	v_lshl_or_b32 v99, v95, 6, v34
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	s_barrier
	;;#ASMEND
	ds_read_b128 v[2:5], v144
	v_or_b32_e32 v145, v107, v96
	v_or_b32_e32 v146, v99, v96
	v_or_b32_e32 v90, 0x60, v111
	v_lshlrev_b32_e32 v110, 5, v6
	ds_read_b128 v[6:9], v145
	ds_read_b128 v[10:13], v145 offset:1024
	ds_read_b128 v[14:17], v145 offset:2048
	ds_read_b128 v[18:21], v145 offset:3072
	ds_read_b128 v[22:25], v145 offset:4096
	ds_read_b128 v[26:29], v145 offset:5120
	ds_read_b128 v[30:33], v145 offset:6144
	ds_read_b128 v[34:37], v146
	ds_read_b128 v[38:41], v146 offset:1024
	ds_read_b128 v[42:45], v146 offset:2048
	ds_read_b128 v[50:53], v146 offset:3072
	ds_read_b128 v[54:57], v146 offset:4096
	ds_read_b128 v[112:115], v146 offset:5120
	ds_read_b128 v[116:119], v146 offset:6144
	ds_read_b128 v[120:123], v146 offset:7168
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[252:255], v[2:5], v[34:37], 0
	;;#ASMEND
	s_add_i32 s26, s21, 0x1c000
	v_add_lshl_u32 v46, v90, v98, 1
	;;#ASMSTART
	s_mov_b32 m0, s26
	buffer_load_dwordx4 v46, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[124:127], v144 offset:16384
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[248:251], v[2:5], v[38:41], 0
	;;#ASMEND
	s_waitcnt lgkmcnt(6)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[244:247], v[2:5], v[42:45], 0
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[240:243], v[2:5], v[50:53], 0
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[236:239], v[2:5], v[54:57], 0
	;;#ASMEND
	ds_read_b128 v[128:131], v145 offset:16384
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[232:235], v[2:5], v[112:115], 0
	;;#ASMEND
	s_waitcnt lgkmcnt(3)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[228:231], v[2:5], v[116:119], 0
	;;#ASMEND
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[224:227], v[2:5], v[120:123], 0
	;;#ASMEND
	v_add_lshl_u32 v2, v90, v100, 1
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[220:223], v[6:9], v[34:37], 0
	;;#ASMEND
	s_add_i32 s26, s21, 0x1d000
	;;#ASMSTART
	s_mov_b32 m0, s26
	buffer_load_dwordx4 v2, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[2:5], v145 offset:17408
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[216:219], v[6:9], v[38:41], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[212:215], v[6:9], v[42:45], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[208:211], v[6:9], v[50:53], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[204:207], v[6:9], v[54:57], 0
	;;#ASMEND
	ds_read_b128 v[132:135], v145 offset:18432
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[200:203], v[6:9], v[112:115], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[196:199], v[6:9], v[116:119], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[192:195], v[6:9], v[120:123], 0
	;;#ASMEND
	v_add_lshl_u32 v6, v90, v101, 1
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[188:191], v[10:13], v[34:37], 0
	;;#ASMEND
	s_add_i32 s26, s21, 0x1e000
	;;#ASMSTART
	s_mov_b32 m0, s26
	buffer_load_dwordx4 v6, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	v_add_lshl_u32 v6, v90, v102, 1
	ds_read_b128 v[136:139], v145 offset:19456
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[184:187], v[10:13], v[38:41], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[180:183], v[10:13], v[42:45], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[176:179], v[10:13], v[50:53], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[172:175], v[10:13], v[54:57], 0
	;;#ASMEND
	ds_read_b128 v[46:49], v145 offset:20480
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[168:171], v[10:13], v[112:115], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[164:167], v[10:13], v[116:119], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[160:163], v[10:13], v[120:123], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[156:159], v[14:17], v[34:37], 0
	;;#ASMEND
	s_add_i32 s26, s21, 0x1f000
	;;#ASMSTART
	s_mov_b32 m0, s26
	buffer_load_dwordx4 v6, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	v_add_lshl_u32 v6, v103, v90, 1
	ds_read_b128 v[58:61], v145 offset:21504
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[152:155], v[14:17], v[38:41], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[148:151], v[14:17], v[42:45], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[144:147], v[14:17], v[50:53], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[140:143], v[14:17], v[54:57], 0
	;;#ASMEND
	ds_read_b128 v[66:69], v145 offset:22528
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[136:139], v[14:17], v[112:115], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[132:135], v[14:17], v[116:119], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[128:131], v[14:17], v[120:123], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[124:127], v[18:21], v[34:37], 0
	;;#ASMEND
	s_add_i32 s26, s21, 0xc000
	;;#ASMSTART
	s_mov_b32 m0, s26
	buffer_load_dwordx4 v6, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	v_add_lshl_u32 v6, v104, v90, 1
	ds_read_b128 v[70:73], v146 offset:16384
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[120:123], v[18:21], v[38:41], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[116:119], v[18:21], v[42:45], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[112:115], v[18:21], v[50:53], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[108:111], v[18:21], v[54:57], 0
	;;#ASMEND
	ds_read_b128 v[62:65], v146 offset:17408
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[104:107], v[18:21], v[112:115], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[100:103], v[18:21], v[116:119], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[96:99], v[18:21], v[120:123], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[92:95], v[22:25], v[34:37], 0
	;;#ASMEND
	s_add_i32 s26, s21, 0xd000
	;;#ASMSTART
	s_mov_b32 m0, s26
	buffer_load_dwordx4 v6, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	v_add_lshl_u32 v6, v105, v90, 1
	ds_read_b128 v[74:77], v146 offset:18432
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[88:91], v[22:25], v[38:41], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[84:87], v[22:25], v[42:45], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[80:83], v[22:25], v[50:53], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[76:79], v[22:25], v[54:57], 0
	;;#ASMEND
	ds_read_b128 v[78:81], v146 offset:19456
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[72:75], v[22:25], v[112:115], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[68:71], v[22:25], v[116:119], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[64:67], v[22:25], v[120:123], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[60:63], v[26:29], v[34:37], 0
	;;#ASMEND
	s_add_i32 s26, s21, 0xe000
	;;#ASMSTART
	s_mov_b32 m0, s26
	buffer_load_dwordx4 v6, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	v_add_lshl_u32 v6, v106, v90, 1
	v_or_b32_e32 v111, 0x80, v111
	ds_read_b128 v[82:85], v146 offset:20480
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[56:59], v[26:29], v[38:41], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[52:55], v[26:29], v[42:45], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[48:51], v[26:29], v[50:53], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[44:47], v[26:29], v[54:57], 0
	;;#ASMEND
	ds_read_b128 v[86:89], v146 offset:21504
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[40:43], v[26:29], v[112:115], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[36:39], v[26:29], v[116:119], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[32:35], v[26:29], v[120:123], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[28:31], v[30:33], v[34:37], 0
	;;#ASMEND
	s_add_i32 s26, s21, 0xf000
	;;#ASMSTART
	s_mov_b32 m0, s26
	buffer_load_dwordx4 v6, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	v_add_lshl_u32 v6, v111, v98, 1
	ds_read_b128 v[90:93], v146 offset:22528
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[24:27], v[30:33], v[38:41], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[20:23], v[30:33], v[42:45], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[16:19], v[30:33], v[50:53], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[12:15], v[30:33], v[54:57], 0
	;;#ASMEND
	ds_read_b128 v[140:143], v146 offset:23552
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[8:11], v[30:33], v[112:115], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[4:7], v[30:33], v[116:119], 0
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[0:3], v[30:33], v[120:123], 0
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	s_barrier
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[252:255], v[124:127], v[70:73], a[252:255]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s25
	buffer_load_dwordx4 v6, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	v_add_lshl_u32 v6, v111, v100, 1
	ds_read_b128 v[50:53], v144 offset:32768
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[248:251], v[124:127], v[62:65], a[248:251]
	;;#ASMEND
	s_waitcnt lgkmcnt(6)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[244:247], v[124:127], v[74:77], a[244:247]
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[240:243], v[124:127], v[78:81], a[240:243]
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[236:239], v[124:127], v[82:85], a[236:239]
	;;#ASMEND
	ds_read_b128 v[42:45], v145 offset:32768
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[232:235], v[124:127], v[86:89], a[232:235]
	;;#ASMEND
	s_waitcnt lgkmcnt(3)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[228:231], v[124:127], v[90:93], a[228:231]
	;;#ASMEND
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[224:227], v[124:127], v[140:143], a[224:227]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[220:223], v[128:131], v[70:73], a[220:223]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s24
	buffer_load_dwordx4 v6, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[34:37], v145 offset:33792
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[216:219], v[128:131], v[62:65], a[216:219]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[212:215], v[128:131], v[74:77], a[212:215]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[208:211], v[128:131], v[78:81], a[208:211]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[204:207], v[128:131], v[82:85], a[204:207]
	;;#ASMEND
	ds_read_b128 v[26:29], v145 offset:34816
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[200:203], v[128:131], v[86:89], a[200:203]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[196:199], v[128:131], v[90:93], a[196:199]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[192:195], v[128:131], v[140:143], a[192:195]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[188:191], v[2:5], v[70:73], a[188:191]
	;;#ASMEND
	v_add_lshl_u32 v6, v111, v101, 1
	;;#ASMSTART
	s_mov_b32 m0, s23
	buffer_load_dwordx4 v6, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[14:17], v145 offset:35840
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[184:187], v[2:5], v[62:65], a[184:187]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[180:183], v[2:5], v[74:77], a[180:183]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[176:179], v[2:5], v[78:81], a[176:179]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[172:175], v[2:5], v[82:85], a[172:175]
	;;#ASMEND
	ds_read_b128 v[10:13], v145 offset:36864
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[168:171], v[2:5], v[86:89], a[168:171]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[164:167], v[2:5], v[90:93], a[164:167]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[160:163], v[2:5], v[140:143], a[160:163]
	;;#ASMEND
	v_add_lshl_u32 v2, v111, v102, 1
	v_add_lshl_u32 v18, v103, v111, 1
	v_add_lshl_u32 v30, v104, v111, 1
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[156:159], v[132:135], v[70:73], a[156:159]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s17
	buffer_load_dwordx4 v2, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[6:9], v145 offset:37888
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[152:155], v[132:135], v[62:65], a[152:155]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[148:151], v[132:135], v[74:77], a[148:151]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[144:147], v[132:135], v[78:81], a[144:147]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[140:143], v[132:135], v[82:85], a[140:143]
	;;#ASMEND
	ds_read_b128 v[2:5], v145 offset:38912
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[136:139], v[132:135], v[86:89], a[136:139]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[132:135], v[132:135], v[90:93], a[132:135]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[128:131], v[132:135], v[140:143], a[128:131]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[124:127], v[136:139], v[70:73], a[124:127]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s21
	buffer_load_dwordx4 v18, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[18:21], v146 offset:32768
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[120:123], v[136:139], v[62:65], a[120:123]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[116:119], v[136:139], v[74:77], a[116:119]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[112:115], v[136:139], v[78:81], a[112:115]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[108:111], v[136:139], v[82:85], a[108:111]
	;;#ASMEND
	ds_read_b128 v[22:25], v146 offset:33792
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[104:107], v[136:139], v[86:89], a[104:107]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[100:103], v[136:139], v[90:93], a[100:103]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[96:99], v[136:139], v[140:143], a[96:99]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[92:95], v[46:49], v[70:73], a[92:95]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s16
	buffer_load_dwordx4 v30, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[30:33], v146 offset:34816
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[88:91], v[46:49], v[62:65], a[88:91]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[84:87], v[46:49], v[74:77], a[84:87]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[80:83], v[46:49], v[78:81], a[80:83]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[76:79], v[46:49], v[82:85], a[76:79]
	;;#ASMEND
	ds_read_b128 v[38:41], v146 offset:35840
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[72:75], v[46:49], v[86:89], a[72:75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[68:71], v[46:49], v[90:93], a[68:71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[64:67], v[46:49], v[140:143], a[64:67]
	;;#ASMEND
	v_add_lshl_u32 v46, v105, v111, 1
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[60:63], v[58:61], v[70:73], a[60:63]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s15
	buffer_load_dwordx4 v46, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[46:49], v146 offset:36864
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[56:59], v[58:61], v[62:65], a[56:59]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[52:55], v[58:61], v[74:77], a[52:55]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[48:51], v[58:61], v[78:81], a[48:51]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[44:47], v[58:61], v[82:85], a[44:47]
	;;#ASMEND
	ds_read_b128 v[54:57], v146 offset:37888
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[40:43], v[58:61], v[86:89], a[40:43]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[36:39], v[58:61], v[90:93], a[36:39]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[32:35], v[58:61], v[140:143], a[32:35]
	;;#ASMEND
	v_add_lshl_u32 v58, v106, v111, 1
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[28:31], v[66:69], v[70:73], a[28:31]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s14
	buffer_load_dwordx4 v58, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[58:61], v146 offset:38912
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[24:27], v[66:69], v[62:65], a[24:27]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[20:23], v[66:69], v[74:77], a[20:23]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[16:19], v[66:69], v[78:81], a[16:19]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[12:15], v[66:69], v[82:85], a[12:15]
	;;#ASMEND
	ds_read_b128 v[62:65], v146 offset:39936
	v_lshlrev_b32_e32 v71, 5, v95
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[8:11], v[66:69], v[86:89], a[8:11]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[4:7], v[66:69], v[90:93], a[4:7]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[0:3], v[66:69], v[140:143], a[0:3]
	;;#ASMEND
	v_bitop3_b32 v66, v109, 3, v0 bitop3:0x48
	v_lshlrev_b32_e32 v66, 4, v66
	s_or_b32 s23, s2, 0x3fc0
	s_or_b32 s24, s2, 0xc0
	v_lshl_or_b32 v67, s3, 15, v66
	s_movk_i32 s2, 0x8100
	v_lshl_add_u32 v66, v102, 1, v67
	v_lshl_add_u32 v68, v106, 1, v67
	v_lshl_add_u32 v70, v101, 1, v67
	v_lshl_add_u32 v72, v105, 1, v67
	v_lshl_add_u32 v74, v100, 1, v67
	v_lshl_add_u32 v76, v104, 1, v67
	v_lshl_add_u32 v78, v103, 1, v67
	v_lshl_add_u32 v80, v98, 1, v67
	s_mov_b64 s[14:15], 1
	s_mov_b32 s3, -1
	v_lshlrev_b32_e32 v67, 1, v110
	v_lshlrev_b32_e32 v69, 1, v71
	;;#ASMSTART
	s_waitcnt vmcnt(16)
	s_barrier
	;;#ASMEND
.LBB0_1:
	s_mov_b64 s[16:17], s[14:15]
	s_min_i32 s15, s24, s23
	v_add_u32_e32 v82, s2, v72
	v_add_u32_e32 v83, s2, v68
	s_xor_b32 s14, s16, 1
	s_lshl_b32 s16, s16, 15
	v_add_u32_e32 v109, 0x8040, v82
	v_add_u32_e32 v154, 0x8040, v83
	v_or_b32_e32 v82, s15, v97
	s_lshl_b32 s17, s14, 15
	v_or_b32_e32 v83, s16, v96
	v_add_u32_e32 v71, s2, v80
	v_add_u32_e32 v73, s2, v74
	v_add_u32_e32 v75, s2, v70
	v_add_u32_e32 v77, s2, v66
	v_add_u32_e32 v79, s2, v78
	v_add_u32_e32 v81, s2, v76
	s_add_i32 s15, s16, s21
	v_add_lshl_u32 v162, v82, v98, 1
	v_add_lshl_u32 v163, v82, v100, 1
	v_add_lshl_u32 v164, v82, v101, 1
	v_add_lshl_u32 v165, v82, v102, 1
	v_add_lshl_u32 v166, v82, v103, 1
	v_add_lshl_u32 v167, v82, v104, 1
	v_add_lshl_u32 v168, v82, v105, 1
	v_add_lshl_u32 v169, v82, v106, 1
	s_add_i32 s16, s21, s17
	v_add_u32_e32 v82, v83, v108
	v_add_u32_e32 v126, v83, v67
	v_add3_u32 v158, v83, v69, s22
	v_or_b32_e32 v83, s17, v96
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[252:255], v[50:53], v[18:21], a[252:255]
	;;#ASMEND
	v_add_u32_e32 v71, 0x8040, v71
	v_add_u32_e32 v73, 0x8040, v73
	v_add_u32_e32 v75, 0x8040, v75
	v_add_u32_e32 v77, 0x8040, v77
	v_add_u32_e32 v79, 0x8040, v79
	v_add_u32_e32 v81, 0x8040, v81
	s_add_i32 s31, s16, 0x4000
	s_add_i32 s33, s16, 0x14000
	s_add_i32 s34, s16, 0x5000
	s_add_i32 s35, s16, 0x15000
	s_add_i32 s36, s16, 0x6000
	s_add_i32 s37, s16, 0x16000
	s_add_i32 s38, s16, 0x7000
	s_add_i32 s16, s16, 0x17000
	v_add_u32_e32 v170, v83, v108
	v_add_u32_e32 v171, v83, v67
	v_add3_u32 v172, v83, v69, s22
	;;#ASMSTART
	s_mov_b32 m0, s33
	buffer_load_dwordx4 v71, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[82:85], v82 offset:16384
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[248:251], v[50:53], v[22:25], a[248:251]
	;;#ASMEND
	s_waitcnt lgkmcnt(6)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[244:247], v[50:53], v[30:33], a[244:247]
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[240:243], v[50:53], v[38:41], a[240:243]
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[236:239], v[50:53], v[46:49], a[236:239]
	;;#ASMEND
	ds_read_b128 v[86:89], v126 offset:16384
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[232:235], v[50:53], v[54:57], a[232:235]
	;;#ASMEND
	s_waitcnt lgkmcnt(3)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[228:231], v[50:53], v[58:61], a[228:231]
	;;#ASMEND
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[224:227], v[50:53], v[62:65], a[224:227]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[220:223], v[42:45], v[18:21], a[220:223]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s35
	buffer_load_dwordx4 v73, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[90:93], v126 offset:17408
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[216:219], v[42:45], v[22:25], a[216:219]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[212:215], v[42:45], v[30:33], a[212:215]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[208:211], v[42:45], v[38:41], a[208:211]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[204:207], v[42:45], v[46:49], a[204:207]
	;;#ASMEND
	ds_read_b128 v[110:113], v126 offset:18432
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[200:203], v[42:45], v[54:57], a[200:203]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[196:199], v[42:45], v[58:61], a[196:199]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[192:195], v[42:45], v[62:65], a[192:195]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[188:191], v[34:37], v[18:21], a[188:191]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s37
	buffer_load_dwordx4 v75, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[114:117], v126 offset:19456
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[184:187], v[34:37], v[22:25], a[184:187]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[180:183], v[34:37], v[30:33], a[180:183]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[176:179], v[34:37], v[38:41], a[176:179]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[172:175], v[34:37], v[46:49], a[172:175]
	;;#ASMEND
	ds_read_b128 v[118:121], v126 offset:20480
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[168:171], v[34:37], v[54:57], a[168:171]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[164:167], v[34:37], v[58:61], a[164:167]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[160:163], v[34:37], v[62:65], a[160:163]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[156:159], v[26:29], v[18:21], a[156:159]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s16
	buffer_load_dwordx4 v77, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[122:125], v126 offset:21504
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[152:155], v[26:29], v[22:25], a[152:155]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[148:151], v[26:29], v[30:33], a[148:151]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[144:147], v[26:29], v[38:41], a[144:147]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[140:143], v[26:29], v[46:49], a[140:143]
	;;#ASMEND
	ds_read_b128 v[126:129], v126 offset:22528
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[136:139], v[26:29], v[54:57], a[136:139]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[132:135], v[26:29], v[58:61], a[132:135]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[128:131], v[26:29], v[62:65], a[128:131]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[124:127], v[14:17], v[18:21], a[124:127]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s31
	buffer_load_dwordx4 v79, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[130:133], v158 offset:16384
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[120:123], v[14:17], v[22:25], a[120:123]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[116:119], v[14:17], v[30:33], a[116:119]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[112:115], v[14:17], v[38:41], a[112:115]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[108:111], v[14:17], v[46:49], a[108:111]
	;;#ASMEND
	ds_read_b128 v[134:137], v158 offset:17408
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[104:107], v[14:17], v[54:57], a[104:107]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[100:103], v[14:17], v[58:61], a[100:103]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[96:99], v[14:17], v[62:65], a[96:99]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[92:95], v[10:13], v[18:21], a[92:95]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s34
	buffer_load_dwordx4 v81, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[138:141], v158 offset:18432
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[88:91], v[10:13], v[22:25], a[88:91]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[84:87], v[10:13], v[30:33], a[84:87]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[80:83], v[10:13], v[38:41], a[80:83]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[76:79], v[10:13], v[46:49], a[76:79]
	;;#ASMEND
	ds_read_b128 v[142:145], v158 offset:19456
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[72:75], v[10:13], v[54:57], a[72:75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[68:71], v[10:13], v[58:61], a[68:71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[64:67], v[10:13], v[62:65], a[64:67]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[60:63], v[6:9], v[18:21], a[60:63]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s36
	buffer_load_dwordx4 v109, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[146:149], v158 offset:20480
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[56:59], v[6:9], v[22:25], a[56:59]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[52:55], v[6:9], v[30:33], a[52:55]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[48:51], v[6:9], v[38:41], a[48:51]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[44:47], v[6:9], v[46:49], a[44:47]
	;;#ASMEND
	ds_read_b128 v[150:153], v158 offset:21504
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[40:43], v[6:9], v[54:57], a[40:43]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[36:39], v[6:9], v[58:61], a[36:39]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[32:35], v[6:9], v[62:65], a[32:35]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[28:31], v[2:5], v[18:21], a[28:31]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s38
	buffer_load_dwordx4 v154, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[154:157], v158 offset:22528
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[24:27], v[2:5], v[22:25], a[24:27]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[20:23], v[2:5], v[30:33], a[20:23]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[16:19], v[2:5], v[38:41], a[16:19]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[12:15], v[2:5], v[46:49], a[12:15]
	;;#ASMEND
	s_add_i32 s25, s15, 0x10000
	s_add_i32 s17, s15, 0x11000
	s_add_i32 s26, s15, 0x12000
	s_add_i32 s27, s15, 0x13000
	s_add_i32 s28, s15, 0x1000
	s_add_i32 s29, s15, 0x2000
	s_add_i32 s30, s15, 0x3000
	ds_read_b128 v[158:161], v158 offset:23552
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[8:11], v[2:5], v[54:57], a[8:11]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[4:7], v[2:5], v[58:61], a[4:7]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[0:3], v[2:5], v[62:65], a[0:3]
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	s_barrier
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[252:255], v[82:85], v[130:133], a[252:255]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s25
	buffer_load_dwordx4 v162, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[50:53], v170
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[248:251], v[82:85], v[134:137], a[248:251]
	;;#ASMEND
	s_waitcnt lgkmcnt(6)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[244:247], v[82:85], v[138:141], a[244:247]
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[240:243], v[82:85], v[142:145], a[240:243]
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[236:239], v[82:85], v[146:149], a[236:239]
	;;#ASMEND
	ds_read_b128 v[42:45], v171
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[232:235], v[82:85], v[150:153], a[232:235]
	;;#ASMEND
	s_waitcnt lgkmcnt(3)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[228:231], v[82:85], v[154:157], a[228:231]
	;;#ASMEND
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[224:227], v[82:85], v[158:161], a[224:227]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[220:223], v[86:89], v[130:133], a[220:223]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s17
	buffer_load_dwordx4 v163, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[34:37], v171 offset:1024
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[216:219], v[86:89], v[134:137], a[216:219]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[212:215], v[86:89], v[138:141], a[212:215]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[208:211], v[86:89], v[142:145], a[208:211]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[204:207], v[86:89], v[146:149], a[204:207]
	;;#ASMEND
	ds_read_b128 v[26:29], v171 offset:2048
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[200:203], v[86:89], v[150:153], a[200:203]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[196:199], v[86:89], v[154:157], a[196:199]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[192:195], v[86:89], v[158:161], a[192:195]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[188:191], v[90:93], v[130:133], a[188:191]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s26
	buffer_load_dwordx4 v164, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[14:17], v171 offset:3072
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[184:187], v[90:93], v[134:137], a[184:187]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[180:183], v[90:93], v[138:141], a[180:183]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[176:179], v[90:93], v[142:145], a[176:179]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[172:175], v[90:93], v[146:149], a[172:175]
	;;#ASMEND
	ds_read_b128 v[10:13], v171 offset:4096
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[168:171], v[90:93], v[150:153], a[168:171]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[164:167], v[90:93], v[154:157], a[164:167]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[160:163], v[90:93], v[158:161], a[160:163]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[156:159], v[110:113], v[130:133], a[156:159]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v165, s[8:11], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[6:9], v171 offset:5120
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[152:155], v[110:113], v[134:137], a[152:155]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[148:151], v[110:113], v[138:141], a[148:151]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[144:147], v[110:113], v[142:145], a[144:147]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[140:143], v[110:113], v[146:149], a[140:143]
	;;#ASMEND
	ds_read_b128 v[2:5], v171 offset:6144
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[136:139], v[110:113], v[150:153], a[136:139]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[132:135], v[110:113], v[154:157], a[132:135]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[128:131], v[110:113], v[158:161], a[128:131]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[124:127], v[114:117], v[130:133], a[124:127]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s15
	buffer_load_dwordx4 v166, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[18:21], v172
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[120:123], v[114:117], v[134:137], a[120:123]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[116:119], v[114:117], v[138:141], a[116:119]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[112:115], v[114:117], v[142:145], a[112:115]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[108:111], v[114:117], v[146:149], a[108:111]
	;;#ASMEND
	ds_read_b128 v[22:25], v172 offset:1024
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[104:107], v[114:117], v[150:153], a[104:107]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[100:103], v[114:117], v[154:157], a[100:103]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[96:99], v[114:117], v[158:161], a[96:99]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[92:95], v[118:121], v[130:133], a[92:95]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s28
	buffer_load_dwordx4 v167, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[30:33], v172 offset:2048
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[88:91], v[118:121], v[134:137], a[88:91]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[84:87], v[118:121], v[138:141], a[84:87]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[80:83], v[118:121], v[142:145], a[80:83]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[76:79], v[118:121], v[146:149], a[76:79]
	;;#ASMEND
	ds_read_b128 v[38:41], v172 offset:3072
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[72:75], v[118:121], v[150:153], a[72:75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[68:71], v[118:121], v[154:157], a[68:71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[64:67], v[118:121], v[158:161], a[64:67]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[60:63], v[122:125], v[130:133], a[60:63]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s29
	buffer_load_dwordx4 v168, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[46:49], v172 offset:4096
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[56:59], v[122:125], v[134:137], a[56:59]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[52:55], v[122:125], v[138:141], a[52:55]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[48:51], v[122:125], v[142:145], a[48:51]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[44:47], v[122:125], v[146:149], a[44:47]
	;;#ASMEND
	ds_read_b128 v[54:57], v172 offset:5120
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[40:43], v[122:125], v[150:153], a[40:43]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[36:39], v[122:125], v[154:157], a[36:39]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[32:35], v[122:125], v[158:161], a[32:35]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[28:31], v[126:129], v[130:133], a[28:31]
	;;#ASMEND
	;;#ASMSTART
	s_mov_b32 m0, s30
	buffer_load_dwordx4 v169, s[4:7], 0 offen sc0 lds
	;;#ASMEND
	ds_read_b128 v[58:61], v172 offset:6144
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[24:27], v[126:129], v[134:137], a[24:27]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[20:23], v[126:129], v[138:141], a[20:23]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[16:19], v[126:129], v[142:145], a[16:19]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[12:15], v[126:129], v[146:149], a[12:15]
	;;#ASMEND
	ds_read_b128 v[62:65], v172 offset:7168
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[8:11], v[126:129], v[150:153], a[8:11]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[4:7], v[126:129], v[154:157], a[4:7]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[0:3], v[126:129], v[158:161], a[0:3]
	;;#ASMEND
	s_add_i32 s24, s24, 64
	s_add_u32 s2, s2, 0x80
	s_addc_u32 s3, s3, 0
	s_cmp_lg_u64 s[2:3], 0
	;;#ASMSTART
	s_waitcnt vmcnt(16)
	s_barrier
	;;#ASMEND
	s_cbranch_scc1 .LBB0_1
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	s_lshl_b32 s4, s14, 14
	s_lshl_b32 s4, s4, 1
	v_add3_u32 v67, v108, s4, v96
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[252:255], v[50:53], v[18:21], a[252:255]
	;;#ASMEND
	ds_read_b128 v[68:71], v67 offset:16384
	v_add3_u32 v67, v107, s4, v96
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[248:251], v[50:53], v[22:25], a[248:251]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[244:247], v[50:53], v[30:33], a[244:247]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[240:243], v[50:53], v[38:41], a[240:243]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[236:239], v[50:53], v[46:49], a[236:239]
	;;#ASMEND
	ds_read_b128 v[72:75], v67 offset:16384
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[232:235], v[50:53], v[54:57], a[232:235]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[228:231], v[50:53], v[58:61], a[228:231]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[224:227], v[50:53], v[62:65], a[224:227]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[220:223], v[42:45], v[18:21], a[220:223]
	;;#ASMEND
	ds_read_b128 v[76:79], v67 offset:17408
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[216:219], v[42:45], v[22:25], a[216:219]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[212:215], v[42:45], v[30:33], a[212:215]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[208:211], v[42:45], v[38:41], a[208:211]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[204:207], v[42:45], v[46:49], a[204:207]
	;;#ASMEND
	ds_read_b128 v[80:83], v67 offset:18432
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[200:203], v[42:45], v[54:57], a[200:203]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[196:199], v[42:45], v[58:61], a[196:199]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[192:195], v[42:45], v[62:65], a[192:195]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[188:191], v[34:37], v[18:21], a[188:191]
	;;#ASMEND
	ds_read_b128 v[84:87], v67 offset:19456
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[184:187], v[34:37], v[22:25], a[184:187]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[180:183], v[34:37], v[30:33], a[180:183]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[176:179], v[34:37], v[38:41], a[176:179]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[172:175], v[34:37], v[46:49], a[172:175]
	;;#ASMEND
	ds_read_b128 v[50:53], v67 offset:20480
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[168:171], v[34:37], v[54:57], a[168:171]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[164:167], v[34:37], v[58:61], a[164:167]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[160:163], v[34:37], v[62:65], a[160:163]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[156:159], v[26:29], v[18:21], a[156:159]
	;;#ASMEND
	ds_read_b128 v[42:45], v67 offset:21504
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[152:155], v[26:29], v[22:25], a[152:155]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[148:151], v[26:29], v[30:33], a[148:151]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[144:147], v[26:29], v[38:41], a[144:147]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[140:143], v[26:29], v[46:49], a[140:143]
	;;#ASMEND
	ds_read_b128 v[34:37], v67 offset:22528
	v_add3_u32 v67, v99, s4, v96
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, -1
	s_and_b32 s1, s1, 0xffff
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[136:139], v[26:29], v[54:57], a[136:139]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[132:135], v[26:29], v[58:61], a[132:135]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[128:131], v[26:29], v[62:65], a[128:131]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[124:127], v[14:17], v[18:21], a[124:127]
	;;#ASMEND
	ds_read_b128 v[26:29], v67 offset:16384
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[120:123], v[14:17], v[22:25], a[120:123]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[116:119], v[14:17], v[30:33], a[116:119]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[112:115], v[14:17], v[38:41], a[112:115]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[108:111], v[14:17], v[46:49], a[108:111]
	;;#ASMEND
	ds_read_b128 v[88:91], v67 offset:17408
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[104:107], v[14:17], v[54:57], a[104:107]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[100:103], v[14:17], v[58:61], a[100:103]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[96:99], v[14:17], v[62:65], a[96:99]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[92:95], v[10:13], v[18:21], a[92:95]
	;;#ASMEND
	ds_read_b128 v[14:17], v67 offset:18432
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[88:91], v[10:13], v[22:25], a[88:91]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[84:87], v[10:13], v[30:33], a[84:87]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[80:83], v[10:13], v[38:41], a[80:83]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[76:79], v[10:13], v[46:49], a[76:79]
	;;#ASMEND
	ds_read_b128 v[96:99], v67 offset:19456
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[72:75], v[10:13], v[54:57], a[72:75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[68:71], v[10:13], v[58:61], a[68:71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[64:67], v[10:13], v[62:65], a[64:67]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[60:63], v[6:9], v[18:21], a[60:63]
	;;#ASMEND
	ds_read_b128 v[10:13], v67 offset:20480
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[56:59], v[6:9], v[22:25], a[56:59]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[52:55], v[6:9], v[30:33], a[52:55]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[48:51], v[6:9], v[38:41], a[48:51]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[44:47], v[6:9], v[46:49], a[44:47]
	;;#ASMEND
	ds_read_b128 v[100:103], v67 offset:21504
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[40:43], v[6:9], v[54:57], a[40:43]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[36:39], v[6:9], v[58:61], a[36:39]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[32:35], v[6:9], v[62:65], a[32:35]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[28:31], v[2:5], v[18:21], a[28:31]
	;;#ASMEND
	ds_read_b128 v[6:9], v67 offset:22528
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[24:27], v[2:5], v[22:25], a[24:27]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[20:23], v[2:5], v[30:33], a[20:23]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[16:19], v[2:5], v[38:41], a[16:19]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[12:15], v[2:5], v[46:49], a[12:15]
	;;#ASMEND
	ds_read_b128 v[18:21], v67 offset:23552
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[8:11], v[2:5], v[54:57], a[8:11]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[4:7], v[2:5], v[58:61], a[4:7]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[0:3], v[2:5], v[62:65], a[0:3]
	;;#ASMEND
	v_or_b32_e32 v66, 0x70, v1
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	s_barrier
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[252:255], v[68:71], v[26:29], a[252:255]
	;;#ASMEND
	s_waitcnt lgkmcnt(6)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[248:251], v[68:71], v[88:91], a[248:251]
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[244:247], v[68:71], v[14:17], a[244:247]
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[240:243], v[68:71], v[96:99], a[240:243]
	;;#ASMEND
	s_waitcnt lgkmcnt(3)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[236:239], v[68:71], v[10:13], a[236:239]
	;;#ASMEND
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[232:235], v[68:71], v[100:103], a[232:235]
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[228:231], v[68:71], v[6:9], a[228:231]
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[224:227], v[68:71], v[18:21], a[224:227]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[220:223], v[72:75], v[26:29], a[220:223]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[216:219], v[72:75], v[88:91], a[216:219]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[212:215], v[72:75], v[14:17], a[212:215]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[208:211], v[72:75], v[96:99], a[208:211]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[204:207], v[72:75], v[10:13], a[204:207]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[200:203], v[72:75], v[100:103], a[200:203]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[196:199], v[72:75], v[6:9], a[196:199]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[192:195], v[72:75], v[18:21], a[192:195]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[188:191], v[76:79], v[26:29], a[188:191]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[184:187], v[76:79], v[88:91], a[184:187]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[180:183], v[76:79], v[14:17], a[180:183]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[176:179], v[76:79], v[96:99], a[176:179]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[172:175], v[76:79], v[10:13], a[172:175]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[168:171], v[76:79], v[100:103], a[168:171]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[164:167], v[76:79], v[6:9], a[164:167]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[160:163], v[76:79], v[18:21], a[160:163]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[156:159], v[80:83], v[26:29], a[156:159]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[152:155], v[80:83], v[88:91], a[152:155]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[148:151], v[80:83], v[14:17], a[148:151]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[144:147], v[80:83], v[96:99], a[144:147]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[140:143], v[80:83], v[10:13], a[140:143]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[136:139], v[80:83], v[100:103], a[136:139]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[132:135], v[80:83], v[6:9], a[132:135]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[128:131], v[80:83], v[18:21], a[128:131]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[124:127], v[84:87], v[26:29], a[124:127]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[120:123], v[84:87], v[88:91], a[120:123]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[116:119], v[84:87], v[14:17], a[116:119]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[112:115], v[84:87], v[96:99], a[112:115]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[108:111], v[84:87], v[10:13], a[108:111]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[104:107], v[84:87], v[100:103], a[104:107]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[100:103], v[84:87], v[6:9], a[100:103]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[96:99], v[84:87], v[18:21], a[96:99]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[92:95], v[50:53], v[26:29], a[92:95]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[88:91], v[50:53], v[88:91], a[88:91]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[84:87], v[50:53], v[14:17], a[84:87]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[80:83], v[50:53], v[96:99], a[80:83]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[76:79], v[50:53], v[10:13], a[76:79]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[72:75], v[50:53], v[100:103], a[72:75]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[68:71], v[50:53], v[6:9], a[68:71]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[64:67], v[50:53], v[18:21], a[64:67]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[60:63], v[42:45], v[26:29], a[60:63]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[56:59], v[42:45], v[88:91], a[56:59]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[52:55], v[42:45], v[14:17], a[52:55]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[48:51], v[42:45], v[96:99], a[48:51]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[44:47], v[42:45], v[10:13], a[44:47]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[40:43], v[42:45], v[100:103], a[40:43]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[36:39], v[42:45], v[6:9], a[36:39]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[32:35], v[42:45], v[18:21], a[32:35]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[28:31], v[34:37], v[26:29], a[28:31]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[24:27], v[34:37], v[88:91], a[24:27]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[20:23], v[34:37], v[14:17], a[20:23]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[16:19], v[34:37], v[96:99], a[16:19]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[12:15], v[34:37], v[10:13], a[12:15]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[8:11], v[34:37], v[100:103], a[8:11]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[4:7], v[34:37], v[6:9], a[4:7]
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 a[0:3], v[34:37], v[18:21], a[0:3]
	;;#ASMEND
	v_lshlrev_b32_e32 v2, 2, v94
	v_accvgpr_read_b32 v3, a252
	v_and_or_b32 v4, v2, 12, v1
	v_cvt_pk_bf16_f32 v5, v3, s0
	v_lshlrev_b32_e32 v3, 1, v95
	v_lshl_or_b32 v4, v4, 9, v3
	s_barrier
	ds_write_b16 v4, v5
	v_accvgpr_read_b32 v5, a253
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:512
	v_accvgpr_read_b32 v5, a254
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:1024
	v_accvgpr_read_b32 v5, a255
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:1536
	v_accvgpr_read_b32 v5, a248
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:32
	v_accvgpr_read_b32 v5, a249
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:544
	v_accvgpr_read_b32 v5, a250
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:1056
	v_accvgpr_read_b32 v5, a251
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:1568
	v_accvgpr_read_b32 v5, a244
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:64
	v_accvgpr_read_b32 v5, a245
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:576
	v_accvgpr_read_b32 v5, a246
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:1088
	v_accvgpr_read_b32 v5, a247
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:1600
	v_accvgpr_read_b32 v5, a240
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:96
	v_accvgpr_read_b32 v5, a241
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:608
	v_accvgpr_read_b32 v5, a242
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:1120
	v_accvgpr_read_b32 v5, a243
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:1632
	v_accvgpr_read_b32 v5, a236
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:128
	v_accvgpr_read_b32 v5, a237
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:640
	v_accvgpr_read_b32 v5, a238
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:1152
	v_accvgpr_read_b32 v5, a239
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:1664
	v_accvgpr_read_b32 v5, a232
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:160
	v_accvgpr_read_b32 v5, a233
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:672
	v_accvgpr_read_b32 v5, a234
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:1184
	v_accvgpr_read_b32 v5, a235
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:1696
	v_accvgpr_read_b32 v5, a228
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:192
	v_accvgpr_read_b32 v5, a229
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:704
	v_accvgpr_read_b32 v5, a230
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:1216
	v_accvgpr_read_b32 v5, a231
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:1728
	v_accvgpr_read_b32 v5, a224
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:224
	v_accvgpr_read_b32 v5, a225
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:736
	v_accvgpr_read_b32 v5, a226
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:1248
	v_accvgpr_read_b32 v5, a227
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:1760
	v_accvgpr_read_b32 v5, a220
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:8192
	v_accvgpr_read_b32 v5, a221
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:8704
	v_accvgpr_read_b32 v5, a222
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:9216
	v_accvgpr_read_b32 v5, a223
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:9728
	v_accvgpr_read_b32 v5, a216
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:8224
	v_accvgpr_read_b32 v5, a217
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:8736
	v_accvgpr_read_b32 v5, a218
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:9248
	v_accvgpr_read_b32 v5, a219
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:9760
	v_accvgpr_read_b32 v5, a212
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:8256
	v_accvgpr_read_b32 v5, a213
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:8768
	v_accvgpr_read_b32 v5, a214
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:9280
	v_accvgpr_read_b32 v5, a215
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:9792
	v_accvgpr_read_b32 v5, a208
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:8288
	v_accvgpr_read_b32 v5, a209
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:8800
	v_accvgpr_read_b32 v5, a210
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:9312
	v_accvgpr_read_b32 v5, a211
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:9824
	v_accvgpr_read_b32 v5, a204
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:8320
	v_accvgpr_read_b32 v5, a205
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:8832
	v_accvgpr_read_b32 v5, a206
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:9344
	v_accvgpr_read_b32 v5, a207
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:9856
	v_accvgpr_read_b32 v5, a200
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:8352
	v_accvgpr_read_b32 v5, a201
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:8864
	v_accvgpr_read_b32 v5, a202
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:9376
	v_accvgpr_read_b32 v5, a203
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:9888
	v_accvgpr_read_b32 v5, a196
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:8384
	v_accvgpr_read_b32 v5, a197
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:8896
	v_accvgpr_read_b32 v5, a198
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:9408
	v_accvgpr_read_b32 v5, a199
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:9920
	v_accvgpr_read_b32 v5, a192
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:8416
	v_accvgpr_read_b32 v5, a193
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:8928
	v_accvgpr_read_b32 v5, a194
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:9440
	v_accvgpr_read_b32 v5, a195
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:9952
	v_accvgpr_read_b32 v5, a188
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:16384
	v_accvgpr_read_b32 v5, a189
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:16896
	v_accvgpr_read_b32 v5, a190
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:17408
	v_accvgpr_read_b32 v5, a191
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:17920
	v_accvgpr_read_b32 v5, a184
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:16416
	v_accvgpr_read_b32 v5, a185
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:16928
	v_accvgpr_read_b32 v5, a186
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:17440
	v_accvgpr_read_b32 v5, a187
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:17952
	v_accvgpr_read_b32 v5, a180
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:16448
	v_accvgpr_read_b32 v5, a181
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:16960
	v_accvgpr_read_b32 v5, a182
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:17472
	v_accvgpr_read_b32 v5, a183
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:17984
	v_accvgpr_read_b32 v5, a176
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:16480
	v_accvgpr_read_b32 v5, a177
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:16992
	v_accvgpr_read_b32 v5, a178
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:17504
	v_accvgpr_read_b32 v5, a179
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:18016
	v_accvgpr_read_b32 v5, a172
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:16512
	v_accvgpr_read_b32 v5, a173
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:17024
	v_accvgpr_read_b32 v5, a174
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:17536
	v_accvgpr_read_b32 v5, a175
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:18048
	v_accvgpr_read_b32 v5, a168
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:16544
	v_accvgpr_read_b32 v5, a169
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:17056
	v_accvgpr_read_b32 v5, a170
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:17568
	v_accvgpr_read_b32 v5, a171
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:18080
	v_accvgpr_read_b32 v5, a164
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:16576
	v_accvgpr_read_b32 v5, a165
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:17088
	v_accvgpr_read_b32 v5, a166
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:17600
	v_accvgpr_read_b32 v5, a167
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:18112
	v_accvgpr_read_b32 v5, a160
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:16608
	v_accvgpr_read_b32 v5, a161
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:17120
	v_accvgpr_read_b32 v5, a162
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:17632
	v_accvgpr_read_b32 v5, a163
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v4, v5 offset:18144
	v_or3_b32 v1, v1, 48, v2
	v_accvgpr_read_b32 v5, a156
	v_cvt_pk_bf16_f32 v5, v5, s0
	v_lshl_or_b32 v1, v1, 9, v3
	ds_write_b16 v1, v5
	v_accvgpr_read_b32 v5, a157
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:512
	v_accvgpr_read_b32 v5, a158
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:1024
	v_accvgpr_read_b32 v5, a159
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:1536
	v_accvgpr_read_b32 v5, a152
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:32
	v_accvgpr_read_b32 v5, a153
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:544
	v_accvgpr_read_b32 v5, a154
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:1056
	v_accvgpr_read_b32 v5, a155
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:1568
	v_accvgpr_read_b32 v5, a148
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:64
	v_accvgpr_read_b32 v5, a149
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:576
	v_accvgpr_read_b32 v5, a150
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:1088
	v_accvgpr_read_b32 v5, a151
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:1600
	v_accvgpr_read_b32 v5, a144
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:96
	v_accvgpr_read_b32 v5, a145
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:608
	v_accvgpr_read_b32 v5, a146
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:1120
	v_accvgpr_read_b32 v5, a147
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:1632
	v_accvgpr_read_b32 v5, a140
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:128
	v_accvgpr_read_b32 v5, a141
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:640
	v_accvgpr_read_b32 v5, a142
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:1152
	v_accvgpr_read_b32 v5, a143
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:1664
	v_accvgpr_read_b32 v5, a136
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:160
	v_accvgpr_read_b32 v5, a137
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:672
	v_accvgpr_read_b32 v5, a138
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:1184
	v_accvgpr_read_b32 v5, a139
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:1696
	v_accvgpr_read_b32 v5, a132
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:192
	v_accvgpr_read_b32 v5, a133
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:704
	v_accvgpr_read_b32 v5, a134
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:1216
	v_accvgpr_read_b32 v5, a135
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:1728
	v_accvgpr_read_b32 v5, a128
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:224
	v_accvgpr_read_b32 v5, a129
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:736
	v_accvgpr_read_b32 v5, a130
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:1248
	v_accvgpr_read_b32 v5, a131
	v_cvt_pk_bf16_f32 v5, v5, s0
	ds_write_b16 v1, v5 offset:1760
	v_accvgpr_read_b32 v1, a124
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:32768
	v_accvgpr_read_b32 v1, a125
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:33280
	v_accvgpr_read_b32 v1, a126
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:33792
	v_accvgpr_read_b32 v1, a127
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:34304
	v_accvgpr_read_b32 v1, a120
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:32800
	v_accvgpr_read_b32 v1, a121
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:33312
	v_accvgpr_read_b32 v1, a122
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:33824
	v_accvgpr_read_b32 v1, a123
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:34336
	v_accvgpr_read_b32 v1, a116
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:32832
	v_accvgpr_read_b32 v1, a117
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:33344
	v_accvgpr_read_b32 v1, a118
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:33856
	v_accvgpr_read_b32 v1, a119
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:34368
	v_accvgpr_read_b32 v1, a112
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:32864
	v_accvgpr_read_b32 v1, a113
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:33376
	v_accvgpr_read_b32 v1, a114
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:33888
	v_accvgpr_read_b32 v1, a115
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:34400
	v_accvgpr_read_b32 v1, a108
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:32896
	v_accvgpr_read_b32 v1, a109
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:33408
	v_accvgpr_read_b32 v1, a110
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:33920
	v_accvgpr_read_b32 v1, a111
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:34432
	v_accvgpr_read_b32 v1, a104
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:32928
	v_accvgpr_read_b32 v1, a105
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:33440
	v_accvgpr_read_b32 v1, a106
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:33952
	v_accvgpr_read_b32 v1, a107
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:34464
	v_accvgpr_read_b32 v1, a100
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:32960
	v_accvgpr_read_b32 v1, a101
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:33472
	v_accvgpr_read_b32 v1, a102
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:33984
	v_accvgpr_read_b32 v1, a103
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:34496
	v_accvgpr_read_b32 v1, a96
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:32992
	v_accvgpr_read_b32 v1, a97
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:33504
	v_accvgpr_read_b32 v1, a98
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:34016
	v_accvgpr_read_b32 v1, a99
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:34528
	v_accvgpr_read_b32 v1, a92
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:40960
	v_accvgpr_read_b32 v1, a93
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:41472
	v_accvgpr_read_b32 v1, a94
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:41984
	v_accvgpr_read_b32 v1, a95
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:42496
	v_accvgpr_read_b32 v1, a88
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:40992
	v_accvgpr_read_b32 v1, a89
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:41504
	v_accvgpr_read_b32 v1, a90
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:42016
	v_accvgpr_read_b32 v1, a91
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:42528
	v_accvgpr_read_b32 v1, a84
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:41024
	v_accvgpr_read_b32 v1, a85
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:41536
	v_accvgpr_read_b32 v1, a86
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:42048
	v_accvgpr_read_b32 v1, a87
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:42560
	v_accvgpr_read_b32 v1, a80
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:41056
	v_accvgpr_read_b32 v1, a81
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:41568
	v_accvgpr_read_b32 v1, a82
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:42080
	v_accvgpr_read_b32 v1, a83
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:42592
	v_accvgpr_read_b32 v1, a76
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:41088
	v_accvgpr_read_b32 v1, a77
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:41600
	v_accvgpr_read_b32 v1, a78
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:42112
	v_accvgpr_read_b32 v1, a79
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:42624
	v_accvgpr_read_b32 v1, a72
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:41120
	v_accvgpr_read_b32 v1, a73
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:41632
	v_accvgpr_read_b32 v1, a74
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:42144
	v_accvgpr_read_b32 v1, a75
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:42656
	v_accvgpr_read_b32 v1, a68
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:41152
	v_accvgpr_read_b32 v1, a69
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:41664
	v_accvgpr_read_b32 v1, a70
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:42176
	v_accvgpr_read_b32 v1, a71
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:42688
	v_accvgpr_read_b32 v1, a64
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:41184
	v_accvgpr_read_b32 v1, a65
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:41696
	v_accvgpr_read_b32 v1, a66
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:42208
	v_accvgpr_read_b32 v1, a67
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:42720
	v_accvgpr_read_b32 v1, a60
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:49152
	v_accvgpr_read_b32 v1, a61
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:49664
	v_accvgpr_read_b32 v1, a62
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:50176
	v_accvgpr_read_b32 v1, a63
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:50688
	v_accvgpr_read_b32 v1, a56
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:49184
	v_accvgpr_read_b32 v1, a57
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:49696
	v_accvgpr_read_b32 v1, a58
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:50208
	v_accvgpr_read_b32 v1, a59
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:50720
	v_accvgpr_read_b32 v1, a52
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:49216
	v_accvgpr_read_b32 v1, a53
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:49728
	v_accvgpr_read_b32 v1, a54
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:50240
	v_accvgpr_read_b32 v1, a55
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:50752
	v_accvgpr_read_b32 v1, a48
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:49248
	v_accvgpr_read_b32 v1, a49
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:49760
	v_accvgpr_read_b32 v1, a50
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:50272
	v_accvgpr_read_b32 v1, a51
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:50784
	v_accvgpr_read_b32 v1, a44
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:49280
	v_accvgpr_read_b32 v1, a45
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:49792
	v_accvgpr_read_b32 v1, a46
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:50304
	v_accvgpr_read_b32 v1, a47
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:50816
	v_accvgpr_read_b32 v1, a40
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:49312
	v_accvgpr_read_b32 v1, a41
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:49824
	v_accvgpr_read_b32 v1, a42
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:50336
	v_accvgpr_read_b32 v1, a43
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:50848
	v_accvgpr_read_b32 v1, a36
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:49344
	v_accvgpr_read_b32 v1, a37
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:49856
	v_accvgpr_read_b32 v1, a38
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:50368
	v_accvgpr_read_b32 v1, a39
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:50880
	v_accvgpr_read_b32 v1, a32
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:49376
	v_accvgpr_read_b32 v1, a33
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:49888
	v_accvgpr_read_b32 v1, a34
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:50400
	v_accvgpr_read_b32 v1, a35
	v_cvt_pk_bf16_f32 v1, v1, s0
	ds_write_b16 v4, v1 offset:50912
	v_or_b32_e32 v1, v2, v66
	v_accvgpr_read_b32 v2, a28
	v_cvt_pk_bf16_f32 v2, v2, s0
	v_lshl_or_b32 v1, v1, 9, v3
	ds_write_b16 v1, v2
	v_accvgpr_read_b32 v2, a29
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:512
	v_accvgpr_read_b32 v2, a30
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:1024
	v_accvgpr_read_b32 v2, a31
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:1536
	v_accvgpr_read_b32 v2, a24
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:32
	v_accvgpr_read_b32 v2, a25
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:544
	v_accvgpr_read_b32 v2, a26
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:1056
	v_accvgpr_read_b32 v2, a27
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:1568
	v_accvgpr_read_b32 v2, a20
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:64
	v_accvgpr_read_b32 v2, a21
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:576
	v_accvgpr_read_b32 v2, a22
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:1088
	v_accvgpr_read_b32 v2, a23
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:1600
	v_accvgpr_read_b32 v2, a16
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:96
	v_accvgpr_read_b32 v2, a17
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:608
	v_accvgpr_read_b32 v2, a18
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:1120
	v_accvgpr_read_b32 v2, a19
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:1632
	v_accvgpr_read_b32 v2, a12
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:128
	v_accvgpr_read_b32 v2, a13
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:640
	v_accvgpr_read_b32 v2, a14
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:1152
	v_accvgpr_read_b32 v2, a15
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:1664
	v_accvgpr_read_b32 v2, a8
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:160
	v_accvgpr_read_b32 v2, a9
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:672
	v_accvgpr_read_b32 v2, a10
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:1184
	v_accvgpr_read_b32 v2, a11
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:1696
	v_accvgpr_read_b32 v2, a4
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:192
	v_accvgpr_read_b32 v2, a5
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:704
	v_accvgpr_read_b32 v2, a6
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:1216
	v_accvgpr_read_b32 v2, a7
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:1728
	v_accvgpr_read_b32 v2, a0
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:224
	v_accvgpr_read_b32 v2, a1
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:736
	v_accvgpr_read_b32 v2, a2
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v1, v2 offset:1248
	v_accvgpr_read_b32 v2, a3
	v_cvt_pk_bf16_f32 v2, v2, s0
	v_lshrrev_b32_e32 v4, 5, v0
	ds_write_b16 v1, v2 offset:1760
	v_or_b32_e32 v2, s20, v4
	v_mov_b32_e32 v3, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[2:3]
	v_lshlrev_b32_e32 v3, 3, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_4
	v_and_b32_e32 v0, 0xf8, v3
	v_lshlrev_b32_e32 v1, 1, v0
	v_lshl_or_b32 v1, v4, 9, v1
	ds_read_b128 v[6:9], v1
	v_or_b32_e32 v0, s18, v0
	v_lshlrev_b32_e32 v1, 14, v2
	v_lshl_add_u32 v0, v0, 1, v1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_4:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 8, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_6
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_6:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 16, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_8
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_8:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 24, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_10
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_10:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 32, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_12
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_12:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 40, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_14
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_14:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 48, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_16
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_16:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 56, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_18
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_18:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 64, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_20
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_20:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0x48, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_22
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_22:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0x50, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_24
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_24:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0x58, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_26
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_26:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0x60, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_28
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_28:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0x68, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_30
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_30:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0x70, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_32
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_32:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0x78, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_34
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_34:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0x80, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_36
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_36:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0x88, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_38
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_38:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0x90, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_40
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_40:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0x98, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_42
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_42:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0xa0, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_44
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_44:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0xa8, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_46
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_46:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0xb0, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_48
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_48:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0xb8, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_50
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_50:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0xc0, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_52
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_52:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0xc8, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_54
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_54:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0xd0, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_56
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_56:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0xd8, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_58
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_58:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0xe0, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_60
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_60:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0xe8, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_62
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_62:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0xf0, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_64
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v5, 1, v1
	v_lshl_or_b32 v2, v2, 9, v5
	ds_read_b128 v[6:9], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v0, s[0:3], 0 offen
.LBB0_64:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v2, 0xf8, v4
	v_or_b32_e32 v0, s20, v2
	v_mov_b32_e32 v1, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_66
	v_and_b32_e32 v1, 0xf8, v3
	v_lshlrev_b32_e32 v3, 1, v1
	v_lshl_or_b32 v2, v2, 9, v3
	ds_read_b128 v[2:5], v2
	v_or_b32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v0, 14, v0
	v_lshl_add_u32 v0, v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[2:5], v0, s[0:3], 0 offen
.LBB0_66:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0
		.amdhsa_group_segment_fixed_size 131072
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 132
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 432
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 176
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0, .Lfunc_end0-hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0

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
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     256
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           16
        .value_kind:     by_value
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           16
        .value_kind:     by_value
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           16
        .value_kind:     by_value
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
      - .offset:         80
        .size:           16
        .value_kind:     by_value
      - .offset:         96
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         104
        .size:           8
        .value_kind:     global_buffer
      - .offset:         112
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     global_buffer
      - .offset:         128
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 131072
    .kernarg_segment_align: 8
    .kernarg_segment_size: 132
    .max_flat_workgroup_size: 256
    .name:           hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     45
    .sgpr_spill_count: 0
    .symbol:         hgemm_bf16_256x256x64x2_SPK1_W2x2x1_BLDS1_TN_AS1_AK32_BK32_RP1_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     432
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
