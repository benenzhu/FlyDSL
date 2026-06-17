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
	v_lshrrev_b32_e32 v002, 6, v000
	s_subb_u32 s14, s18, 0
	s_lshl_b32 s20, s14, 8
	v_readfirstlane_b32 s14, v002
	v_lshrrev_b32_e32 v109, 2, v000
	s_sub_i32 s15, s2, s19
	s_lshl_b32 s21, s14, 10
	v_xor_b32_e32 v002, v109, v000
	s_lshl_b32 s2, s3, 14
	s_ashr_i32 s19, s20, 31
	s_lshl_b32 s18, s15, 8
	v_lshlrev_b32_e32 v002, 3, v002
	s_add_i32 s25, s21, 0x10000
	v_and_b32_e32 v097, 24, v002
	v_or_b32_e32 v002, s18, v109
	s_cmpk_lt_u32 s18, 0x2000
	v_lshlrev_b32_e32 v002, 14, v002
	s_cselect_b64 vcc, -1, 0
	v_or_b32_e32 v111, s2, v097
	v_cndmask_b32_e32 v098, 0, v002, vcc
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, -1
	v_add_lshl_u32 v002, v111, v098, 1
	v_or_b32_e32 v004, 64, v109
	s_mov_b32 s10, s6
	s_mov_b32 s11, s7
	s_mov_b32 m0, s25
	buffer_load_dwordx4 v002, s[8:11], 0 offen sc0 lds
	v_or_b32_e32 v002, s18, v004
	v_lshlrev_b32_e32 v002, 14, v002
	v_cndmask_b32_e32 v100, 0, v002, vcc
	v_add_lshl_u32 v002, v111, v100, 1
	v_or_b32_e32 v005, 0x80, v109
	s_add_i32 s24, s21, 0x11000
	s_mov_b32 m0, s24
	buffer_load_dwordx4 v002, s[8:11], 0 offen sc0 lds
	v_or_b32_e32 v002, s18, v005
	v_lshlrev_b32_e32 v002, 14, v002
	v_cndmask_b32_e32 v101, 0, v002, vcc
	v_add_lshl_u32 v002, v111, v101, 1
	v_or_b32_e32 v006, 0xc0, v109
	s_add_i32 s23, s21, 0x12000
	s_mov_b32 m0, s23
	buffer_load_dwordx4 v002, s[8:11], 0 offen sc0 lds
	v_or_b32_e32 v002, s18, v006
	v_lshlrev_b32_e32 v002, 14, v002
	v_cndmask_b32_e32 v102, 0, v002, vcc
	v_add_lshl_u32 v002, v111, v102, 1
	v_or_b32_e32 v007, 32, v111
	s_add_i32 s17, s21, 0x13000
	s_mov_b32 m0, s17
	buffer_load_dwordx4 v002, s[8:11], 0 offen sc0 lds
	v_add_lshl_u32 v002, v007, v098, 1
	s_add_i32 s14, s21, 0x14000
	s_mov_b32 m0, s14
	buffer_load_dwordx4 v002, s[8:11], 0 offen sc0 lds
	v_add_lshl_u32 v002, v007, v100, 1
	s_add_i32 s14, s21, 0x15000
	s_mov_b32 m0, s14
	buffer_load_dwordx4 v002, s[8:11], 0 offen sc0 lds
	v_add_lshl_u32 v002, v007, v101, 1
	s_add_i32 s14, s21, 0x16000
	s_mov_b32 m0, s14
	buffer_load_dwordx4 v002, s[8:11], 0 offen sc0 lds
	v_add_lshl_u32 v002, v007, v102, 1
	s_add_i32 s14, s21, 0x17000
	s_mov_b32 m0, s14
	buffer_load_dwordx4 v002, s[8:11], 0 offen sc0 lds
	v_or_b32_e32 v002, s20, v109
	v_mov_b32_e32 v003, s19
	v_lshlrev_b32_e32 v008, 14, v002
	v_cmp_gt_u64_e32 vcc, s[12:13], v002 v003
	s_add_i32 s16, s21, 0x1000
	s_add_i32 s15, s21, 0x2000
	v_cndmask_b32_e32 v103, 0, v008, vcc
	v_add_lshl_u32 v002, v103, v111, 1
	s_mov_b32 m0, s21
	buffer_load_dwordx4 v002, s[4:7], 0 offen sc0 lds
	v_or_b32_e32 v002, s20, v004
	v_lshlrev_b32_e32 v004, 14, v002
	v_cmp_gt_u64_e32 vcc, s[12:13], v002 v003
	s_add_i32 s14, s21, 0x3000
	s_add_i32 s27, s21, 0x4000
	v_cndmask_b32_e32 v104, 0, v004, vcc
	v_add_lshl_u32 v002, v104, v111, 1
	s_mov_b32 m0, s16
	buffer_load_dwordx4 v002, s[4:7], 0 offen sc0 lds
	v_or_b32_e32 v002, s20, v005
	v_lshlrev_b32_e32 v004, 14, v002
	v_cmp_gt_u64_e32 vcc, s[12:13], v002 v003
	v_and_b32_e32 v001, 0x80, v000
	v_lshlrev_b32_e32 v034, 1, v000
	v_cndmask_b32_e32 v105, 0, v004, vcc
	v_add_lshl_u32 v002, v105, v111, 1
	s_mov_b32 m0, s15
	buffer_load_dwordx4 v002, s[4:7], 0 offen sc0 lds
	v_or_b32_e32 v002, s20, v006
	v_lshlrev_b32_e32 v004, 14, v002
	v_cmp_gt_u64_e32 vcc, s[12:13], v002 v003
	v_and_b32_e32 v035, 15, v000
	v_lshrrev_b32_e32 v094, 4, v000
	v_cndmask_b32_e32 v106, 0, v004, vcc
	v_add_lshl_u32 v002, v106, v111, 1
	s_mov_b32 m0, s14
	buffer_load_dwordx4 v002, s[4:7], 0 offen sc0 lds
	v_add_lshl_u32 v002, v103, v007, 1
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v002, s[4:7], 0 offen sc0 lds
	v_add_lshl_u32 v002, v104, v007, 1
	s_add_i32 s27, s21, 0x5000
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v002, s[4:7], 0 offen sc0 lds
	v_add_lshl_u32 v002, v105, v007, 1
	s_add_i32 s27, s21, 0x6000
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v002, s[4:7], 0 offen sc0 lds
	v_add_lshl_u32 v002, v106, v007, 1
	s_add_i32 s27, s21, 0x7000
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v002, s[4:7], 0 offen sc0 lds
	v_or_b32_e32 v002, 64, v111
	s_add_i32 s27, s21, 0x18000
	v_add_lshl_u32 v003, v002, v098, 1
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v003, s[8:11], 0 offen sc0 lds
	s_add_i32 s27, s21, 0x19000
	v_add_lshl_u32 v003, v002, v100, 1
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v003, s[8:11], 0 offen sc0 lds
	s_add_i32 s27, s21, 0x1a000
	v_add_lshl_u32 v003, v002, v101, 1
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v003, s[8:11], 0 offen sc0 lds
	s_add_i32 s27, s21, 0x1b000
	v_add_lshl_u32 v003, v002, v102, 1
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v003, s[8:11], 0 offen sc0 lds
	s_add_i32 s27, s21, 0x8000
	v_add_lshl_u32 v003, v103, v002, 1
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v003, s[4:7], 0 offen sc0 lds
	s_add_i32 s27, s21, 0x9000
	v_add_lshl_u32 v003, v104, v002, 1
	s_mov_b32 s22, 0x10000
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v003, s[4:7], 0 offen sc0 lds
	s_add_i32 s27, s21, 0xa000
	v_add_lshl_u32 v003, v105, v002, 1
	v_add_lshl_u32 v002, v106, v002, 1
	s_movk_i32 s26, 0x80
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v003, s[4:7], 0 offen sc0 lds
	s_add_i32 s27, s21, 0xb000
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v002, s[4:7], 0 offen sc0 lds
	v_xor_b32_e32 v002, v094, v000
	v_lshlrev_b32_e32 v002, 4, v002
	v_and_b32_e32 v096, 48, v002
	v_lshlrev_b32_e32 v002, 5, v000
	v_and_b32_e32 v002, 0x11e0, v002
	v_lshlrev_b32_e32 v108, 1, v002
	v_or3_b32 v006, v001, v035, 16
	v_and_or_b32 v095, v034, s26, v035
	v_mov_b32_e32 v034, 0x10000
	v_or_b32_e32 v144, v108, v096
	v_lshlrev_b32_e32 v107, 6, v006
	v_lshl_or_b32 v099, v095, 6, v034
	s_waitcnt vmcnt(0)
	s_barrier
	ds_read_b128 v002 v003 v004 v005, v144
	v_or_b32_e32 v145, v107, v096
	v_or_b32_e32 v146, v099, v096
	v_or_b32_e32 v090, 0x60, v111
	v_lshlrev_b32_e32 v110, 5, v006
	ds_read_b128 v006 v007 v008 v009, v145
	ds_read_b128 v010 v011 v012 v013, v145 offset:1024
	ds_read_b128 v014 v015 v016 v017, v145 offset:2048
	ds_read_b128 v018 v019 v020 v021, v145 offset:3072
	ds_read_b128 v022 v023 v024 v025, v145 offset:4096
	ds_read_b128 v026 v027 v028 v029, v145 offset:5120
	ds_read_b128 v030 v031 v032 v033, v145 offset:6144
	ds_read_b128 v034 v035 v036 v037, v146
	ds_read_b128 v038 v039 v040 v041, v146 offset:1024
	ds_read_b128 v042 v043 v044 v045, v146 offset:2048
	ds_read_b128 v050 v051 v052 v053, v146 offset:3072
	ds_read_b128 v054 v055 v056 v057, v146 offset:4096
	ds_read_b128 v112 v113 v114 v115, v146 offset:5120
	ds_read_b128 v116 v117 v118 v119, v146 offset:6144
	ds_read_b128 v120 v121 v122 v123, v146 offset:7168
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_bf16 a[252:255], v002 v003 v004 v005, v034 v035 v036 v037, 0
	s_add_i32 s26, s21, 0x1c000
	v_add_lshl_u32 v046, v090, v098, 1
	s_mov_b32 m0, s26
	buffer_load_dwordx4 v046, s[8:11], 0 offen sc0 lds
	ds_read_b128 v124 v125 v126 v127, v144 offset:16384
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_bf16 a[248:251], v002 v003 v004 v005, v038 v039 v040 v041, 0
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 a[244:247], v002 v003 v004 v005, v042 v043 v044 v045, 0
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 a[240:243], v002 v003 v004 v005, v050 v051 v052 v053, 0
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 a[236:239], v002 v003 v004 v005, v054 v055 v056 v057, 0
	ds_read_b128 v128 v129 v130 v131, v145 offset:16384
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 a[232:235], v002 v003 v004 v005, v112 v113 v114 v115, 0
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 a[228:231], v002 v003 v004 v005, v116 v117 v118 v119, 0
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 a[224:227], v002 v003 v004 v005, v120 v121 v122 v123, 0
	v_add_lshl_u32 v002, v090, v100, 1
	v_mfma_f32_16x16x32_bf16 a[220:223], v006 v007 v008 v009, v034 v035 v036 v037, 0
	s_add_i32 s26, s21, 0x1d000
	s_mov_b32 m0, s26
	buffer_load_dwordx4 v002, s[8:11], 0 offen sc0 lds
	ds_read_b128 v002 v003 v004 v005, v145 offset:17408
	v_mfma_f32_16x16x32_bf16 a[216:219], v006 v007 v008 v009, v038 v039 v040 v041, 0
	v_mfma_f32_16x16x32_bf16 a[212:215], v006 v007 v008 v009, v042 v043 v044 v045, 0
	v_mfma_f32_16x16x32_bf16 a[208:211], v006 v007 v008 v009, v050 v051 v052 v053, 0
	v_mfma_f32_16x16x32_bf16 a[204:207], v006 v007 v008 v009, v054 v055 v056 v057, 0
	ds_read_b128 v132 v133 v134 v135, v145 offset:18432
	v_mfma_f32_16x16x32_bf16 a[200:203], v006 v007 v008 v009, v112 v113 v114 v115, 0
	v_mfma_f32_16x16x32_bf16 a[196:199], v006 v007 v008 v009, v116 v117 v118 v119, 0
	v_mfma_f32_16x16x32_bf16 a[192:195], v006 v007 v008 v009, v120 v121 v122 v123, 0
	v_add_lshl_u32 v006, v090, v101, 1
	v_mfma_f32_16x16x32_bf16 a[188:191], v010 v011 v012 v013, v034 v035 v036 v037, 0
	s_add_i32 s26, s21, 0x1e000
	s_mov_b32 m0, s26
	buffer_load_dwordx4 v006, s[8:11], 0 offen sc0 lds
	v_add_lshl_u32 v006, v090, v102, 1
	ds_read_b128 v136 v137 v138 v139, v145 offset:19456
	v_mfma_f32_16x16x32_bf16 a[184:187], v010 v011 v012 v013, v038 v039 v040 v041, 0
	v_mfma_f32_16x16x32_bf16 a[180:183], v010 v011 v012 v013, v042 v043 v044 v045, 0
	v_mfma_f32_16x16x32_bf16 a[176:179], v010 v011 v012 v013, v050 v051 v052 v053, 0
	v_mfma_f32_16x16x32_bf16 a[172:175], v010 v011 v012 v013, v054 v055 v056 v057, 0
	ds_read_b128 v046 v047 v048 v049, v145 offset:20480
	v_mfma_f32_16x16x32_bf16 a[168:171], v010 v011 v012 v013, v112 v113 v114 v115, 0
	v_mfma_f32_16x16x32_bf16 a[164:167], v010 v011 v012 v013, v116 v117 v118 v119, 0
	v_mfma_f32_16x16x32_bf16 a[160:163], v010 v011 v012 v013, v120 v121 v122 v123, 0
	v_mfma_f32_16x16x32_bf16 a[156:159], v014 v015 v016 v017, v034 v035 v036 v037, 0
	s_add_i32 s26, s21, 0x1f000
	s_mov_b32 m0, s26
	buffer_load_dwordx4 v006, s[8:11], 0 offen sc0 lds
	v_add_lshl_u32 v006, v103, v090, 1
	ds_read_b128 v058 v059 v060 v061, v145 offset:21504
	v_mfma_f32_16x16x32_bf16 a[152:155], v014 v015 v016 v017, v038 v039 v040 v041, 0
	v_mfma_f32_16x16x32_bf16 a[148:151], v014 v015 v016 v017, v042 v043 v044 v045, 0
	v_mfma_f32_16x16x32_bf16 a[144:147], v014 v015 v016 v017, v050 v051 v052 v053, 0
	v_mfma_f32_16x16x32_bf16 a[140:143], v014 v015 v016 v017, v054 v055 v056 v057, 0
	ds_read_b128 v066 v067 v068 v069, v145 offset:22528
	v_mfma_f32_16x16x32_bf16 a[136:139], v014 v015 v016 v017, v112 v113 v114 v115, 0
	v_mfma_f32_16x16x32_bf16 a[132:135], v014 v015 v016 v017, v116 v117 v118 v119, 0
	v_mfma_f32_16x16x32_bf16 a[128:131], v014 v015 v016 v017, v120 v121 v122 v123, 0
	v_mfma_f32_16x16x32_bf16 a[124:127], v018 v019 v020 v021, v034 v035 v036 v037, 0
	s_add_i32 s26, s21, 0xc000
	s_mov_b32 m0, s26
	buffer_load_dwordx4 v006, s[4:7], 0 offen sc0 lds
	v_add_lshl_u32 v006, v104, v090, 1
	ds_read_b128 v070 v071 v072 v073, v146 offset:16384
	v_mfma_f32_16x16x32_bf16 a[120:123], v018 v019 v020 v021, v038 v039 v040 v041, 0
	v_mfma_f32_16x16x32_bf16 a[116:119], v018 v019 v020 v021, v042 v043 v044 v045, 0
	v_mfma_f32_16x16x32_bf16 a[112:115], v018 v019 v020 v021, v050 v051 v052 v053, 0
	v_mfma_f32_16x16x32_bf16 a[108:111], v018 v019 v020 v021, v054 v055 v056 v057, 0
	ds_read_b128 v062 v063 v064 v065, v146 offset:17408
	v_mfma_f32_16x16x32_bf16 a[104:107], v018 v019 v020 v021, v112 v113 v114 v115, 0
	v_mfma_f32_16x16x32_bf16 a[100:103], v018 v019 v020 v021, v116 v117 v118 v119, 0
	v_mfma_f32_16x16x32_bf16 a[96:99], v018 v019 v020 v021, v120 v121 v122 v123, 0
	v_mfma_f32_16x16x32_bf16 a[92:95], v022 v023 v024 v025, v034 v035 v036 v037, 0
	s_add_i32 s26, s21, 0xd000
	s_mov_b32 m0, s26
	buffer_load_dwordx4 v006, s[4:7], 0 offen sc0 lds
	v_add_lshl_u32 v006, v105, v090, 1
	ds_read_b128 v074 v075 v076 v077, v146 offset:18432
	v_mfma_f32_16x16x32_bf16 a[88:91], v022 v023 v024 v025, v038 v039 v040 v041, 0
	v_mfma_f32_16x16x32_bf16 a[84:87], v022 v023 v024 v025, v042 v043 v044 v045, 0
	v_mfma_f32_16x16x32_bf16 a[80:83], v022 v023 v024 v025, v050 v051 v052 v053, 0
	v_mfma_f32_16x16x32_bf16 a[76:79], v022 v023 v024 v025, v054 v055 v056 v057, 0
	ds_read_b128 v078 v079 v080 v081, v146 offset:19456
	v_mfma_f32_16x16x32_bf16 a[72:75], v022 v023 v024 v025, v112 v113 v114 v115, 0
	v_mfma_f32_16x16x32_bf16 a[68:71], v022 v023 v024 v025, v116 v117 v118 v119, 0
	v_mfma_f32_16x16x32_bf16 a[64:67], v022 v023 v024 v025, v120 v121 v122 v123, 0
	v_mfma_f32_16x16x32_bf16 a[60:63], v026 v027 v028 v029, v034 v035 v036 v037, 0
	s_add_i32 s26, s21, 0xe000
	s_mov_b32 m0, s26
	buffer_load_dwordx4 v006, s[4:7], 0 offen sc0 lds
	v_add_lshl_u32 v006, v106, v090, 1
	v_or_b32_e32 v111, 0x80, v111
	ds_read_b128 v082 v083 v084 v085, v146 offset:20480
	v_mfma_f32_16x16x32_bf16 a[56:59], v026 v027 v028 v029, v038 v039 v040 v041, 0
	v_mfma_f32_16x16x32_bf16 a[52:55], v026 v027 v028 v029, v042 v043 v044 v045, 0
	v_mfma_f32_16x16x32_bf16 a[48:51], v026 v027 v028 v029, v050 v051 v052 v053, 0
	v_mfma_f32_16x16x32_bf16 a[44:47], v026 v027 v028 v029, v054 v055 v056 v057, 0
	ds_read_b128 v086 v087 v088 v089, v146 offset:21504
	v_mfma_f32_16x16x32_bf16 a[40:43], v026 v027 v028 v029, v112 v113 v114 v115, 0
	v_mfma_f32_16x16x32_bf16 a[36:39], v026 v027 v028 v029, v116 v117 v118 v119, 0
	v_mfma_f32_16x16x32_bf16 a[32:35], v026 v027 v028 v029, v120 v121 v122 v123, 0
	v_mfma_f32_16x16x32_bf16 a[28:31], v030 v031 v032 v033, v034 v035 v036 v037, 0
	s_add_i32 s26, s21, 0xf000
	s_mov_b32 m0, s26
	buffer_load_dwordx4 v006, s[4:7], 0 offen sc0 lds
	v_add_lshl_u32 v006, v111, v098, 1
	ds_read_b128 v090 v091 v092 v093, v146 offset:22528
	v_mfma_f32_16x16x32_bf16 a[24:27], v030 v031 v032 v033, v038 v039 v040 v041, 0
	v_mfma_f32_16x16x32_bf16 a[20:23], v030 v031 v032 v033, v042 v043 v044 v045, 0
	v_mfma_f32_16x16x32_bf16 a[16:19], v030 v031 v032 v033, v050 v051 v052 v053, 0
	v_mfma_f32_16x16x32_bf16 a[12:15], v030 v031 v032 v033, v054 v055 v056 v057, 0
	ds_read_b128 v140 v141 v142 v143, v146 offset:23552
	v_mfma_f32_16x16x32_bf16 a[8:11], v030 v031 v032 v033, v112 v113 v114 v115, 0
	v_mfma_f32_16x16x32_bf16 a[4:7], v030 v031 v032 v033, v116 v117 v118 v119, 0
	v_mfma_f32_16x16x32_bf16 a[0:3], v030 v031 v032 v033, v120 v121 v122 v123, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_bf16 a[252:255], v124 v125 v126 v127, v070 v071 v072 v073, a[252:255]
	s_mov_b32 m0, s25
	buffer_load_dwordx4 v006, s[8:11], 0 offen sc0 lds
	v_add_lshl_u32 v006, v111, v100, 1
	ds_read_b128 v050 v051 v052 v053, v144 offset:32768
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_bf16 a[248:251], v124 v125 v126 v127, v062 v063 v064 v065, a[248:251]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 a[244:247], v124 v125 v126 v127, v074 v075 v076 v077, a[244:247]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 a[240:243], v124 v125 v126 v127, v078 v079 v080 v081, a[240:243]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 a[236:239], v124 v125 v126 v127, v082 v083 v084 v085, a[236:239]
	ds_read_b128 v042 v043 v044 v045, v145 offset:32768
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 a[232:235], v124 v125 v126 v127, v086 v087 v088 v089, a[232:235]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 a[228:231], v124 v125 v126 v127, v090 v091 v092 v093, a[228:231]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 a[224:227], v124 v125 v126 v127, v140 v141 v142 v143, a[224:227]
	v_mfma_f32_16x16x32_bf16 a[220:223], v128 v129 v130 v131, v070 v071 v072 v073, a[220:223]
	s_mov_b32 m0, s24
	buffer_load_dwordx4 v006, s[8:11], 0 offen sc0 lds
	ds_read_b128 v034 v035 v036 v037, v145 offset:33792
	v_mfma_f32_16x16x32_bf16 a[216:219], v128 v129 v130 v131, v062 v063 v064 v065, a[216:219]
	v_mfma_f32_16x16x32_bf16 a[212:215], v128 v129 v130 v131, v074 v075 v076 v077, a[212:215]
	v_mfma_f32_16x16x32_bf16 a[208:211], v128 v129 v130 v131, v078 v079 v080 v081, a[208:211]
	v_mfma_f32_16x16x32_bf16 a[204:207], v128 v129 v130 v131, v082 v083 v084 v085, a[204:207]
	ds_read_b128 v026 v027 v028 v029, v145 offset:34816
	v_mfma_f32_16x16x32_bf16 a[200:203], v128 v129 v130 v131, v086 v087 v088 v089, a[200:203]
	v_mfma_f32_16x16x32_bf16 a[196:199], v128 v129 v130 v131, v090 v091 v092 v093, a[196:199]
	v_mfma_f32_16x16x32_bf16 a[192:195], v128 v129 v130 v131, v140 v141 v142 v143, a[192:195]
	v_mfma_f32_16x16x32_bf16 a[188:191], v002 v003 v004 v005, v070 v071 v072 v073, a[188:191]
	v_add_lshl_u32 v006, v111, v101, 1
	s_mov_b32 m0, s23
	buffer_load_dwordx4 v006, s[8:11], 0 offen sc0 lds
	ds_read_b128 v014 v015 v016 v017, v145 offset:35840
	v_mfma_f32_16x16x32_bf16 a[184:187], v002 v003 v004 v005, v062 v063 v064 v065, a[184:187]
	v_mfma_f32_16x16x32_bf16 a[180:183], v002 v003 v004 v005, v074 v075 v076 v077, a[180:183]
	v_mfma_f32_16x16x32_bf16 a[176:179], v002 v003 v004 v005, v078 v079 v080 v081, a[176:179]
	v_mfma_f32_16x16x32_bf16 a[172:175], v002 v003 v004 v005, v082 v083 v084 v085, a[172:175]
	ds_read_b128 v010 v011 v012 v013, v145 offset:36864
	v_mfma_f32_16x16x32_bf16 a[168:171], v002 v003 v004 v005, v086 v087 v088 v089, a[168:171]
	v_mfma_f32_16x16x32_bf16 a[164:167], v002 v003 v004 v005, v090 v091 v092 v093, a[164:167]
	v_mfma_f32_16x16x32_bf16 a[160:163], v002 v003 v004 v005, v140 v141 v142 v143, a[160:163]
	v_add_lshl_u32 v002, v111, v102, 1
	v_add_lshl_u32 v018, v103, v111, 1
	v_add_lshl_u32 v030, v104, v111, 1
	v_mfma_f32_16x16x32_bf16 a[156:159], v132 v133 v134 v135, v070 v071 v072 v073, a[156:159]
	s_mov_b32 m0, s17
	buffer_load_dwordx4 v002, s[8:11], 0 offen sc0 lds
	ds_read_b128 v006 v007 v008 v009, v145 offset:37888
	v_mfma_f32_16x16x32_bf16 a[152:155], v132 v133 v134 v135, v062 v063 v064 v065, a[152:155]
	v_mfma_f32_16x16x32_bf16 a[148:151], v132 v133 v134 v135, v074 v075 v076 v077, a[148:151]
	v_mfma_f32_16x16x32_bf16 a[144:147], v132 v133 v134 v135, v078 v079 v080 v081, a[144:147]
	v_mfma_f32_16x16x32_bf16 a[140:143], v132 v133 v134 v135, v082 v083 v084 v085, a[140:143]
	ds_read_b128 v002 v003 v004 v005, v145 offset:38912
	v_mfma_f32_16x16x32_bf16 a[136:139], v132 v133 v134 v135, v086 v087 v088 v089, a[136:139]
	v_mfma_f32_16x16x32_bf16 a[132:135], v132 v133 v134 v135, v090 v091 v092 v093, a[132:135]
	v_mfma_f32_16x16x32_bf16 a[128:131], v132 v133 v134 v135, v140 v141 v142 v143, a[128:131]
	v_mfma_f32_16x16x32_bf16 a[124:127], v136 v137 v138 v139, v070 v071 v072 v073, a[124:127]
	s_mov_b32 m0, s21
	buffer_load_dwordx4 v018, s[4:7], 0 offen sc0 lds
	ds_read_b128 v018 v019 v020 v021, v146 offset:32768
	v_mfma_f32_16x16x32_bf16 a[120:123], v136 v137 v138 v139, v062 v063 v064 v065, a[120:123]
	v_mfma_f32_16x16x32_bf16 a[116:119], v136 v137 v138 v139, v074 v075 v076 v077, a[116:119]
	v_mfma_f32_16x16x32_bf16 a[112:115], v136 v137 v138 v139, v078 v079 v080 v081, a[112:115]
	v_mfma_f32_16x16x32_bf16 a[108:111], v136 v137 v138 v139, v082 v083 v084 v085, a[108:111]
	ds_read_b128 v022 v023 v024 v025, v146 offset:33792
	v_mfma_f32_16x16x32_bf16 a[104:107], v136 v137 v138 v139, v086 v087 v088 v089, a[104:107]
	v_mfma_f32_16x16x32_bf16 a[100:103], v136 v137 v138 v139, v090 v091 v092 v093, a[100:103]
	v_mfma_f32_16x16x32_bf16 a[96:99], v136 v137 v138 v139, v140 v141 v142 v143, a[96:99]
	v_mfma_f32_16x16x32_bf16 a[92:95], v046 v047 v048 v049, v070 v071 v072 v073, a[92:95]
	s_mov_b32 m0, s16
	buffer_load_dwordx4 v030, s[4:7], 0 offen sc0 lds
	ds_read_b128 v030 v031 v032 v033, v146 offset:34816
	v_mfma_f32_16x16x32_bf16 a[88:91], v046 v047 v048 v049, v062 v063 v064 v065, a[88:91]
	v_mfma_f32_16x16x32_bf16 a[84:87], v046 v047 v048 v049, v074 v075 v076 v077, a[84:87]
	v_mfma_f32_16x16x32_bf16 a[80:83], v046 v047 v048 v049, v078 v079 v080 v081, a[80:83]
	v_mfma_f32_16x16x32_bf16 a[76:79], v046 v047 v048 v049, v082 v083 v084 v085, a[76:79]
	ds_read_b128 v038 v039 v040 v041, v146 offset:35840
	v_mfma_f32_16x16x32_bf16 a[72:75], v046 v047 v048 v049, v086 v087 v088 v089, a[72:75]
	v_mfma_f32_16x16x32_bf16 a[68:71], v046 v047 v048 v049, v090 v091 v092 v093, a[68:71]
	v_mfma_f32_16x16x32_bf16 a[64:67], v046 v047 v048 v049, v140 v141 v142 v143, a[64:67]
	v_add_lshl_u32 v046, v105, v111, 1
	v_mfma_f32_16x16x32_bf16 a[60:63], v058 v059 v060 v061, v070 v071 v072 v073, a[60:63]
	s_mov_b32 m0, s15
	buffer_load_dwordx4 v046, s[4:7], 0 offen sc0 lds
	ds_read_b128 v046 v047 v048 v049, v146 offset:36864
	v_mfma_f32_16x16x32_bf16 a[56:59], v058 v059 v060 v061, v062 v063 v064 v065, a[56:59]
	v_mfma_f32_16x16x32_bf16 a[52:55], v058 v059 v060 v061, v074 v075 v076 v077, a[52:55]
	v_mfma_f32_16x16x32_bf16 a[48:51], v058 v059 v060 v061, v078 v079 v080 v081, a[48:51]
	v_mfma_f32_16x16x32_bf16 a[44:47], v058 v059 v060 v061, v082 v083 v084 v085, a[44:47]
	ds_read_b128 v054 v055 v056 v057, v146 offset:37888
	v_mfma_f32_16x16x32_bf16 a[40:43], v058 v059 v060 v061, v086 v087 v088 v089, a[40:43]
	v_mfma_f32_16x16x32_bf16 a[36:39], v058 v059 v060 v061, v090 v091 v092 v093, a[36:39]
	v_mfma_f32_16x16x32_bf16 a[32:35], v058 v059 v060 v061, v140 v141 v142 v143, a[32:35]
	v_add_lshl_u32 v058, v106, v111, 1
	v_mfma_f32_16x16x32_bf16 a[28:31], v066 v067 v068 v069, v070 v071 v072 v073, a[28:31]
	s_mov_b32 m0, s14
	buffer_load_dwordx4 v058, s[4:7], 0 offen sc0 lds
	ds_read_b128 v058 v059 v060 v061, v146 offset:38912
	v_mfma_f32_16x16x32_bf16 a[24:27], v066 v067 v068 v069, v062 v063 v064 v065, a[24:27]
	v_mfma_f32_16x16x32_bf16 a[20:23], v066 v067 v068 v069, v074 v075 v076 v077, a[20:23]
	v_mfma_f32_16x16x32_bf16 a[16:19], v066 v067 v068 v069, v078 v079 v080 v081, a[16:19]
	v_mfma_f32_16x16x32_bf16 a[12:15], v066 v067 v068 v069, v082 v083 v084 v085, a[12:15]
	ds_read_b128 v062 v063 v064 v065, v146 offset:39936
	v_lshlrev_b32_e32 v071, 5, v095
	v_mfma_f32_16x16x32_bf16 a[8:11], v066 v067 v068 v069, v086 v087 v088 v089, a[8:11]
	v_mfma_f32_16x16x32_bf16 a[4:7], v066 v067 v068 v069, v090 v091 v092 v093, a[4:7]
	v_mfma_f32_16x16x32_bf16 a[0:3], v066 v067 v068 v069, v140 v141 v142 v143, a[0:3]
	v_bitop3_b32 v066, v109, 3, v000 bitop3:0x48
	v_lshlrev_b32_e32 v066, 4, v066
	s_or_b32 s23, s2, 0x3fc0
	s_or_b32 s24, s2, 0xc0
	v_lshl_or_b32 v067, s3, 15, v066
	s_movk_i32 s2, 0x8100
	v_lshl_add_u32 v066, v102, 1, v067
	v_lshl_add_u32 v068, v106, 1, v067
	v_lshl_add_u32 v070, v101, 1, v067
	v_lshl_add_u32 v072, v105, 1, v067
	v_lshl_add_u32 v074, v100, 1, v067
	v_lshl_add_u32 v076, v104, 1, v067
	v_lshl_add_u32 v078, v103, 1, v067
	v_lshl_add_u32 v080, v098, 1, v067
	s_mov_b64 s[14:15], 1
	s_mov_b32 s3, -1
	v_lshlrev_b32_e32 v067, 1, v110
	v_lshlrev_b32_e32 v069, 1, v071
	s_waitcnt vmcnt(16)
	s_barrier
.LBB0_1:
	s_mov_b64 s[16:17], s[14:15]
	s_min_i32 s15, s24, s23
	v_add_u32_e32 v082, s2, v072
	v_add_u32_e32 v083, s2, v068
	s_xor_b32 s14, s16, 1
	s_lshl_b32 s16, s16, 15
	v_add_u32_e32 v109, 0x8040, v082
	v_add_u32_e32 v154, 0x8040, v083
	v_or_b32_e32 v082, s15, v097
	s_lshl_b32 s17, s14, 15
	v_or_b32_e32 v083, s16, v096
	v_add_u32_e32 v071, s2, v080
	v_add_u32_e32 v073, s2, v074
	v_add_u32_e32 v075, s2, v070
	v_add_u32_e32 v077, s2, v066
	v_add_u32_e32 v079, s2, v078
	v_add_u32_e32 v081, s2, v076
	s_add_i32 s15, s16, s21
	v_add_lshl_u32 v162, v082, v098, 1
	v_add_lshl_u32 v163, v082, v100, 1
	v_add_lshl_u32 v164, v082, v101, 1
	v_add_lshl_u32 v165, v082, v102, 1
	v_add_lshl_u32 v166, v082, v103, 1
	v_add_lshl_u32 v167, v082, v104, 1
	v_add_lshl_u32 v168, v082, v105, 1
	v_add_lshl_u32 v169, v082, v106, 1
	s_add_i32 s16, s21, s17
	v_add_u32_e32 v082, v083, v108
	v_add_u32_e32 v126, v083, v067
	v_add3_u32 v158, v083, v069, s22
	v_or_b32_e32 v083, s17, v096
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_bf16 a[252:255], v050 v051 v052 v053, v018 v019 v020 v021, a[252:255]
	v_add_u32_e32 v071, 0x8040, v071
	v_add_u32_e32 v073, 0x8040, v073
	v_add_u32_e32 v075, 0x8040, v075
	v_add_u32_e32 v077, 0x8040, v077
	v_add_u32_e32 v079, 0x8040, v079
	v_add_u32_e32 v081, 0x8040, v081
	s_add_i32 s31, s16, 0x4000
	s_add_i32 s33, s16, 0x14000
	s_add_i32 s34, s16, 0x5000
	s_add_i32 s35, s16, 0x15000
	s_add_i32 s36, s16, 0x6000
	s_add_i32 s37, s16, 0x16000
	s_add_i32 s38, s16, 0x7000
	s_add_i32 s16, s16, 0x17000
	v_add_u32_e32 v170, v083, v108
	v_add_u32_e32 v171, v083, v067
	v_add3_u32 v172, v083, v069, s22
	s_mov_b32 m0, s33
	buffer_load_dwordx4 v071, s[8:11], 0 offen sc0 lds
	ds_read_b128 v082 v083 v084 v085, v082 offset:16384
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_bf16 a[248:251], v050 v051 v052 v053, v022 v023 v024 v025, a[248:251]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 a[244:247], v050 v051 v052 v053, v030 v031 v032 v033, a[244:247]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 a[240:243], v050 v051 v052 v053, v038 v039 v040 v041, a[240:243]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 a[236:239], v050 v051 v052 v053, v046 v047 v048 v049, a[236:239]
	ds_read_b128 v086 v087 v088 v089, v126 offset:16384
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 a[232:235], v050 v051 v052 v053, v054 v055 v056 v057, a[232:235]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 a[228:231], v050 v051 v052 v053, v058 v059 v060 v061, a[228:231]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 a[224:227], v050 v051 v052 v053, v062 v063 v064 v065, a[224:227]
	v_mfma_f32_16x16x32_bf16 a[220:223], v042 v043 v044 v045, v018 v019 v020 v021, a[220:223]
	s_mov_b32 m0, s35
	buffer_load_dwordx4 v073, s[8:11], 0 offen sc0 lds
	ds_read_b128 v090 v091 v092 v093, v126 offset:17408
	v_mfma_f32_16x16x32_bf16 a[216:219], v042 v043 v044 v045, v022 v023 v024 v025, a[216:219]
	v_mfma_f32_16x16x32_bf16 a[212:215], v042 v043 v044 v045, v030 v031 v032 v033, a[212:215]
	v_mfma_f32_16x16x32_bf16 a[208:211], v042 v043 v044 v045, v038 v039 v040 v041, a[208:211]
	v_mfma_f32_16x16x32_bf16 a[204:207], v042 v043 v044 v045, v046 v047 v048 v049, a[204:207]
	ds_read_b128 v110 v111 v112 v113, v126 offset:18432
	v_mfma_f32_16x16x32_bf16 a[200:203], v042 v043 v044 v045, v054 v055 v056 v057, a[200:203]
	v_mfma_f32_16x16x32_bf16 a[196:199], v042 v043 v044 v045, v058 v059 v060 v061, a[196:199]
	v_mfma_f32_16x16x32_bf16 a[192:195], v042 v043 v044 v045, v062 v063 v064 v065, a[192:195]
	v_mfma_f32_16x16x32_bf16 a[188:191], v034 v035 v036 v037, v018 v019 v020 v021, a[188:191]
	s_mov_b32 m0, s37
	buffer_load_dwordx4 v075, s[8:11], 0 offen sc0 lds
	ds_read_b128 v114 v115 v116 v117, v126 offset:19456
	v_mfma_f32_16x16x32_bf16 a[184:187], v034 v035 v036 v037, v022 v023 v024 v025, a[184:187]
	v_mfma_f32_16x16x32_bf16 a[180:183], v034 v035 v036 v037, v030 v031 v032 v033, a[180:183]
	v_mfma_f32_16x16x32_bf16 a[176:179], v034 v035 v036 v037, v038 v039 v040 v041, a[176:179]
	v_mfma_f32_16x16x32_bf16 a[172:175], v034 v035 v036 v037, v046 v047 v048 v049, a[172:175]
	ds_read_b128 v118 v119 v120 v121, v126 offset:20480
	v_mfma_f32_16x16x32_bf16 a[168:171], v034 v035 v036 v037, v054 v055 v056 v057, a[168:171]
	v_mfma_f32_16x16x32_bf16 a[164:167], v034 v035 v036 v037, v058 v059 v060 v061, a[164:167]
	v_mfma_f32_16x16x32_bf16 a[160:163], v034 v035 v036 v037, v062 v063 v064 v065, a[160:163]
	v_mfma_f32_16x16x32_bf16 a[156:159], v026 v027 v028 v029, v018 v019 v020 v021, a[156:159]
	s_mov_b32 m0, s16
	buffer_load_dwordx4 v077, s[8:11], 0 offen sc0 lds
	ds_read_b128 v122 v123 v124 v125, v126 offset:21504
	v_mfma_f32_16x16x32_bf16 a[152:155], v026 v027 v028 v029, v022 v023 v024 v025, a[152:155]
	v_mfma_f32_16x16x32_bf16 a[148:151], v026 v027 v028 v029, v030 v031 v032 v033, a[148:151]
	v_mfma_f32_16x16x32_bf16 a[144:147], v026 v027 v028 v029, v038 v039 v040 v041, a[144:147]
	v_mfma_f32_16x16x32_bf16 a[140:143], v026 v027 v028 v029, v046 v047 v048 v049, a[140:143]
	ds_read_b128 v126 v127 v128 v129, v126 offset:22528
	v_mfma_f32_16x16x32_bf16 a[136:139], v026 v027 v028 v029, v054 v055 v056 v057, a[136:139]
	v_mfma_f32_16x16x32_bf16 a[132:135], v026 v027 v028 v029, v058 v059 v060 v061, a[132:135]
	v_mfma_f32_16x16x32_bf16 a[128:131], v026 v027 v028 v029, v062 v063 v064 v065, a[128:131]
	v_mfma_f32_16x16x32_bf16 a[124:127], v014 v015 v016 v017, v018 v019 v020 v021, a[124:127]
	s_mov_b32 m0, s31
	buffer_load_dwordx4 v079, s[4:7], 0 offen sc0 lds
	ds_read_b128 v130 v131 v132 v133, v158 offset:16384
	v_mfma_f32_16x16x32_bf16 a[120:123], v014 v015 v016 v017, v022 v023 v024 v025, a[120:123]
	v_mfma_f32_16x16x32_bf16 a[116:119], v014 v015 v016 v017, v030 v031 v032 v033, a[116:119]
	v_mfma_f32_16x16x32_bf16 a[112:115], v014 v015 v016 v017, v038 v039 v040 v041, a[112:115]
	v_mfma_f32_16x16x32_bf16 a[108:111], v014 v015 v016 v017, v046 v047 v048 v049, a[108:111]
	ds_read_b128 v134 v135 v136 v137, v158 offset:17408
	v_mfma_f32_16x16x32_bf16 a[104:107], v014 v015 v016 v017, v054 v055 v056 v057, a[104:107]
	v_mfma_f32_16x16x32_bf16 a[100:103], v014 v015 v016 v017, v058 v059 v060 v061, a[100:103]
	v_mfma_f32_16x16x32_bf16 a[96:99], v014 v015 v016 v017, v062 v063 v064 v065, a[96:99]
	v_mfma_f32_16x16x32_bf16 a[92:95], v010 v011 v012 v013, v018 v019 v020 v021, a[92:95]
	s_mov_b32 m0, s34
	buffer_load_dwordx4 v081, s[4:7], 0 offen sc0 lds
	ds_read_b128 v138 v139 v140 v141, v158 offset:18432
	v_mfma_f32_16x16x32_bf16 a[88:91], v010 v011 v012 v013, v022 v023 v024 v025, a[88:91]
	v_mfma_f32_16x16x32_bf16 a[84:87], v010 v011 v012 v013, v030 v031 v032 v033, a[84:87]
	v_mfma_f32_16x16x32_bf16 a[80:83], v010 v011 v012 v013, v038 v039 v040 v041, a[80:83]
	v_mfma_f32_16x16x32_bf16 a[76:79], v010 v011 v012 v013, v046 v047 v048 v049, a[76:79]
	ds_read_b128 v142 v143 v144 v145, v158 offset:19456
	v_mfma_f32_16x16x32_bf16 a[72:75], v010 v011 v012 v013, v054 v055 v056 v057, a[72:75]
	v_mfma_f32_16x16x32_bf16 a[68:71], v010 v011 v012 v013, v058 v059 v060 v061, a[68:71]
	v_mfma_f32_16x16x32_bf16 a[64:67], v010 v011 v012 v013, v062 v063 v064 v065, a[64:67]
	v_mfma_f32_16x16x32_bf16 a[60:63], v006 v007 v008 v009, v018 v019 v020 v021, a[60:63]
	s_mov_b32 m0, s36
	buffer_load_dwordx4 v109, s[4:7], 0 offen sc0 lds
	ds_read_b128 v146 v147 v148 v149, v158 offset:20480
	v_mfma_f32_16x16x32_bf16 a[56:59], v006 v007 v008 v009, v022 v023 v024 v025, a[56:59]
	v_mfma_f32_16x16x32_bf16 a[52:55], v006 v007 v008 v009, v030 v031 v032 v033, a[52:55]
	v_mfma_f32_16x16x32_bf16 a[48:51], v006 v007 v008 v009, v038 v039 v040 v041, a[48:51]
	v_mfma_f32_16x16x32_bf16 a[44:47], v006 v007 v008 v009, v046 v047 v048 v049, a[44:47]
	ds_read_b128 v150 v151 v152 v153, v158 offset:21504
	v_mfma_f32_16x16x32_bf16 a[40:43], v006 v007 v008 v009, v054 v055 v056 v057, a[40:43]
	v_mfma_f32_16x16x32_bf16 a[36:39], v006 v007 v008 v009, v058 v059 v060 v061, a[36:39]
	v_mfma_f32_16x16x32_bf16 a[32:35], v006 v007 v008 v009, v062 v063 v064 v065, a[32:35]
	v_mfma_f32_16x16x32_bf16 a[28:31], v002 v003 v004 v005, v018 v019 v020 v021, a[28:31]
	s_mov_b32 m0, s38
	buffer_load_dwordx4 v154, s[4:7], 0 offen sc0 lds
	ds_read_b128 v154 v155 v156 v157, v158 offset:22528
	v_mfma_f32_16x16x32_bf16 a[24:27], v002 v003 v004 v005, v022 v023 v024 v025, a[24:27]
	v_mfma_f32_16x16x32_bf16 a[20:23], v002 v003 v004 v005, v030 v031 v032 v033, a[20:23]
	v_mfma_f32_16x16x32_bf16 a[16:19], v002 v003 v004 v005, v038 v039 v040 v041, a[16:19]
	v_mfma_f32_16x16x32_bf16 a[12:15], v002 v003 v004 v005, v046 v047 v048 v049, a[12:15]
	s_add_i32 s25, s15, 0x10000
	s_add_i32 s17, s15, 0x11000
	s_add_i32 s26, s15, 0x12000
	s_add_i32 s27, s15, 0x13000
	s_add_i32 s28, s15, 0x1000
	s_add_i32 s29, s15, 0x2000
	s_add_i32 s30, s15, 0x3000
	ds_read_b128 v158 v159 v160 v161, v158 offset:23552
	v_mfma_f32_16x16x32_bf16 a[8:11], v002 v003 v004 v005, v054 v055 v056 v057, a[8:11]
	v_mfma_f32_16x16x32_bf16 a[4:7], v002 v003 v004 v005, v058 v059 v060 v061, a[4:7]
	v_mfma_f32_16x16x32_bf16 a[0:3], v002 v003 v004 v005, v062 v063 v064 v065, a[0:3]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_bf16 a[252:255], v082 v083 v084 v085, v130 v131 v132 v133, a[252:255]
	s_mov_b32 m0, s25
	buffer_load_dwordx4 v162, s[8:11], 0 offen sc0 lds
	ds_read_b128 v050 v051 v052 v053, v170
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_bf16 a[248:251], v082 v083 v084 v085, v134 v135 v136 v137, a[248:251]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 a[244:247], v082 v083 v084 v085, v138 v139 v140 v141, a[244:247]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 a[240:243], v082 v083 v084 v085, v142 v143 v144 v145, a[240:243]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 a[236:239], v082 v083 v084 v085, v146 v147 v148 v149, a[236:239]
	ds_read_b128 v042 v043 v044 v045, v171
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 a[232:235], v082 v083 v084 v085, v150 v151 v152 v153, a[232:235]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 a[228:231], v082 v083 v084 v085, v154 v155 v156 v157, a[228:231]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 a[224:227], v082 v083 v084 v085, v158 v159 v160 v161, a[224:227]
	v_mfma_f32_16x16x32_bf16 a[220:223], v086 v087 v088 v089, v130 v131 v132 v133, a[220:223]
	s_mov_b32 m0, s17
	buffer_load_dwordx4 v163, s[8:11], 0 offen sc0 lds
	ds_read_b128 v034 v035 v036 v037, v171 offset:1024
	v_mfma_f32_16x16x32_bf16 a[216:219], v086 v087 v088 v089, v134 v135 v136 v137, a[216:219]
	v_mfma_f32_16x16x32_bf16 a[212:215], v086 v087 v088 v089, v138 v139 v140 v141, a[212:215]
	v_mfma_f32_16x16x32_bf16 a[208:211], v086 v087 v088 v089, v142 v143 v144 v145, a[208:211]
	v_mfma_f32_16x16x32_bf16 a[204:207], v086 v087 v088 v089, v146 v147 v148 v149, a[204:207]
	ds_read_b128 v026 v027 v028 v029, v171 offset:2048
	v_mfma_f32_16x16x32_bf16 a[200:203], v086 v087 v088 v089, v150 v151 v152 v153, a[200:203]
	v_mfma_f32_16x16x32_bf16 a[196:199], v086 v087 v088 v089, v154 v155 v156 v157, a[196:199]
	v_mfma_f32_16x16x32_bf16 a[192:195], v086 v087 v088 v089, v158 v159 v160 v161, a[192:195]
	v_mfma_f32_16x16x32_bf16 a[188:191], v090 v091 v092 v093, v130 v131 v132 v133, a[188:191]
	s_mov_b32 m0, s26
	buffer_load_dwordx4 v164, s[8:11], 0 offen sc0 lds
	ds_read_b128 v014 v015 v016 v017, v171 offset:3072
	v_mfma_f32_16x16x32_bf16 a[184:187], v090 v091 v092 v093, v134 v135 v136 v137, a[184:187]
	v_mfma_f32_16x16x32_bf16 a[180:183], v090 v091 v092 v093, v138 v139 v140 v141, a[180:183]
	v_mfma_f32_16x16x32_bf16 a[176:179], v090 v091 v092 v093, v142 v143 v144 v145, a[176:179]
	v_mfma_f32_16x16x32_bf16 a[172:175], v090 v091 v092 v093, v146 v147 v148 v149, a[172:175]
	ds_read_b128 v010 v011 v012 v013, v171 offset:4096
	v_mfma_f32_16x16x32_bf16 a[168:171], v090 v091 v092 v093, v150 v151 v152 v153, a[168:171]
	v_mfma_f32_16x16x32_bf16 a[164:167], v090 v091 v092 v093, v154 v155 v156 v157, a[164:167]
	v_mfma_f32_16x16x32_bf16 a[160:163], v090 v091 v092 v093, v158 v159 v160 v161, a[160:163]
	v_mfma_f32_16x16x32_bf16 a[156:159], v110 v111 v112 v113, v130 v131 v132 v133, a[156:159]
	s_mov_b32 m0, s27
	buffer_load_dwordx4 v165, s[8:11], 0 offen sc0 lds
	ds_read_b128 v006 v007 v008 v009, v171 offset:5120
	v_mfma_f32_16x16x32_bf16 a[152:155], v110 v111 v112 v113, v134 v135 v136 v137, a[152:155]
	v_mfma_f32_16x16x32_bf16 a[148:151], v110 v111 v112 v113, v138 v139 v140 v141, a[148:151]
	v_mfma_f32_16x16x32_bf16 a[144:147], v110 v111 v112 v113, v142 v143 v144 v145, a[144:147]
	v_mfma_f32_16x16x32_bf16 a[140:143], v110 v111 v112 v113, v146 v147 v148 v149, a[140:143]
	ds_read_b128 v002 v003 v004 v005, v171 offset:6144
	v_mfma_f32_16x16x32_bf16 a[136:139], v110 v111 v112 v113, v150 v151 v152 v153, a[136:139]
	v_mfma_f32_16x16x32_bf16 a[132:135], v110 v111 v112 v113, v154 v155 v156 v157, a[132:135]
	v_mfma_f32_16x16x32_bf16 a[128:131], v110 v111 v112 v113, v158 v159 v160 v161, a[128:131]
	v_mfma_f32_16x16x32_bf16 a[124:127], v114 v115 v116 v117, v130 v131 v132 v133, a[124:127]
	s_mov_b32 m0, s15
	buffer_load_dwordx4 v166, s[4:7], 0 offen sc0 lds
	ds_read_b128 v018 v019 v020 v021, v172
	v_mfma_f32_16x16x32_bf16 a[120:123], v114 v115 v116 v117, v134 v135 v136 v137, a[120:123]
	v_mfma_f32_16x16x32_bf16 a[116:119], v114 v115 v116 v117, v138 v139 v140 v141, a[116:119]
	v_mfma_f32_16x16x32_bf16 a[112:115], v114 v115 v116 v117, v142 v143 v144 v145, a[112:115]
	v_mfma_f32_16x16x32_bf16 a[108:111], v114 v115 v116 v117, v146 v147 v148 v149, a[108:111]
	ds_read_b128 v022 v023 v024 v025, v172 offset:1024
	v_mfma_f32_16x16x32_bf16 a[104:107], v114 v115 v116 v117, v150 v151 v152 v153, a[104:107]
	v_mfma_f32_16x16x32_bf16 a[100:103], v114 v115 v116 v117, v154 v155 v156 v157, a[100:103]
	v_mfma_f32_16x16x32_bf16 a[96:99], v114 v115 v116 v117, v158 v159 v160 v161, a[96:99]
	v_mfma_f32_16x16x32_bf16 a[92:95], v118 v119 v120 v121, v130 v131 v132 v133, a[92:95]
	s_mov_b32 m0, s28
	buffer_load_dwordx4 v167, s[4:7], 0 offen sc0 lds
	ds_read_b128 v030 v031 v032 v033, v172 offset:2048
	v_mfma_f32_16x16x32_bf16 a[88:91], v118 v119 v120 v121, v134 v135 v136 v137, a[88:91]
	v_mfma_f32_16x16x32_bf16 a[84:87], v118 v119 v120 v121, v138 v139 v140 v141, a[84:87]
	v_mfma_f32_16x16x32_bf16 a[80:83], v118 v119 v120 v121, v142 v143 v144 v145, a[80:83]
	v_mfma_f32_16x16x32_bf16 a[76:79], v118 v119 v120 v121, v146 v147 v148 v149, a[76:79]
	ds_read_b128 v038 v039 v040 v041, v172 offset:3072
	v_mfma_f32_16x16x32_bf16 a[72:75], v118 v119 v120 v121, v150 v151 v152 v153, a[72:75]
	v_mfma_f32_16x16x32_bf16 a[68:71], v118 v119 v120 v121, v154 v155 v156 v157, a[68:71]
	v_mfma_f32_16x16x32_bf16 a[64:67], v118 v119 v120 v121, v158 v159 v160 v161, a[64:67]
	v_mfma_f32_16x16x32_bf16 a[60:63], v122 v123 v124 v125, v130 v131 v132 v133, a[60:63]
	s_mov_b32 m0, s29
	buffer_load_dwordx4 v168, s[4:7], 0 offen sc0 lds
	ds_read_b128 v046 v047 v048 v049, v172 offset:4096
	v_mfma_f32_16x16x32_bf16 a[56:59], v122 v123 v124 v125, v134 v135 v136 v137, a[56:59]
	v_mfma_f32_16x16x32_bf16 a[52:55], v122 v123 v124 v125, v138 v139 v140 v141, a[52:55]
	v_mfma_f32_16x16x32_bf16 a[48:51], v122 v123 v124 v125, v142 v143 v144 v145, a[48:51]
	v_mfma_f32_16x16x32_bf16 a[44:47], v122 v123 v124 v125, v146 v147 v148 v149, a[44:47]
	ds_read_b128 v054 v055 v056 v057, v172 offset:5120
	v_mfma_f32_16x16x32_bf16 a[40:43], v122 v123 v124 v125, v150 v151 v152 v153, a[40:43]
	v_mfma_f32_16x16x32_bf16 a[36:39], v122 v123 v124 v125, v154 v155 v156 v157, a[36:39]
	v_mfma_f32_16x16x32_bf16 a[32:35], v122 v123 v124 v125, v158 v159 v160 v161, a[32:35]
	v_mfma_f32_16x16x32_bf16 a[28:31], v126 v127 v128 v129, v130 v131 v132 v133, a[28:31]
	s_mov_b32 m0, s30
	buffer_load_dwordx4 v169, s[4:7], 0 offen sc0 lds
	ds_read_b128 v058 v059 v060 v061, v172 offset:6144
	v_mfma_f32_16x16x32_bf16 a[24:27], v126 v127 v128 v129, v134 v135 v136 v137, a[24:27]
	v_mfma_f32_16x16x32_bf16 a[20:23], v126 v127 v128 v129, v138 v139 v140 v141, a[20:23]
	v_mfma_f32_16x16x32_bf16 a[16:19], v126 v127 v128 v129, v142 v143 v144 v145, a[16:19]
	v_mfma_f32_16x16x32_bf16 a[12:15], v126 v127 v128 v129, v146 v147 v148 v149, a[12:15]
	ds_read_b128 v062 v063 v064 v065, v172 offset:7168
	v_mfma_f32_16x16x32_bf16 a[8:11], v126 v127 v128 v129, v150 v151 v152 v153, a[8:11]
	v_mfma_f32_16x16x32_bf16 a[4:7], v126 v127 v128 v129, v154 v155 v156 v157, a[4:7]
	v_mfma_f32_16x16x32_bf16 a[0:3], v126 v127 v128 v129, v158 v159 v160 v161, a[0:3]
	s_add_i32 s24, s24, 64
	s_add_u32 s2, s2, 0x80
	s_addc_u32 s3, s3, 0
	s_cmp_lg_u64 s[2:3], 0
	s_waitcnt vmcnt(16)
	s_barrier
.JUMP.LBB0_1:
	s_cbranch_scc1 .LBB0_1
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	s_lshl_b32 s4, s14, 14
	s_lshl_b32 s4, s4, 1
	v_add3_u32 v067, v108, s4, v096
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 a[252:255], v050 v051 v052 v053, v018 v019 v020 v021, a[252:255]
	ds_read_b128 v068 v069 v070 v071, v067 offset:16384
	v_add3_u32 v067, v107, s4, v096
	v_mfma_f32_16x16x32_bf16 a[248:251], v050 v051 v052 v053, v022 v023 v024 v025, a[248:251]
	v_mfma_f32_16x16x32_bf16 a[244:247], v050 v051 v052 v053, v030 v031 v032 v033, a[244:247]
	v_mfma_f32_16x16x32_bf16 a[240:243], v050 v051 v052 v053, v038 v039 v040 v041, a[240:243]
	v_mfma_f32_16x16x32_bf16 a[236:239], v050 v051 v052 v053, v046 v047 v048 v049, a[236:239]
	ds_read_b128 v072 v073 v074 v075, v067 offset:16384
	v_mfma_f32_16x16x32_bf16 a[232:235], v050 v051 v052 v053, v054 v055 v056 v057, a[232:235]
	v_mfma_f32_16x16x32_bf16 a[228:231], v050 v051 v052 v053, v058 v059 v060 v061, a[228:231]
	v_mfma_f32_16x16x32_bf16 a[224:227], v050 v051 v052 v053, v062 v063 v064 v065, a[224:227]
	v_mfma_f32_16x16x32_bf16 a[220:223], v042 v043 v044 v045, v018 v019 v020 v021, a[220:223]
	ds_read_b128 v076 v077 v078 v079, v067 offset:17408
	v_mfma_f32_16x16x32_bf16 a[216:219], v042 v043 v044 v045, v022 v023 v024 v025, a[216:219]
	v_mfma_f32_16x16x32_bf16 a[212:215], v042 v043 v044 v045, v030 v031 v032 v033, a[212:215]
	v_mfma_f32_16x16x32_bf16 a[208:211], v042 v043 v044 v045, v038 v039 v040 v041, a[208:211]
	v_mfma_f32_16x16x32_bf16 a[204:207], v042 v043 v044 v045, v046 v047 v048 v049, a[204:207]
	ds_read_b128 v080 v081 v082 v083, v067 offset:18432
	v_mfma_f32_16x16x32_bf16 a[200:203], v042 v043 v044 v045, v054 v055 v056 v057, a[200:203]
	v_mfma_f32_16x16x32_bf16 a[196:199], v042 v043 v044 v045, v058 v059 v060 v061, a[196:199]
	v_mfma_f32_16x16x32_bf16 a[192:195], v042 v043 v044 v045, v062 v063 v064 v065, a[192:195]
	v_mfma_f32_16x16x32_bf16 a[188:191], v034 v035 v036 v037, v018 v019 v020 v021, a[188:191]
	ds_read_b128 v084 v085 v086 v087, v067 offset:19456
	v_mfma_f32_16x16x32_bf16 a[184:187], v034 v035 v036 v037, v022 v023 v024 v025, a[184:187]
	v_mfma_f32_16x16x32_bf16 a[180:183], v034 v035 v036 v037, v030 v031 v032 v033, a[180:183]
	v_mfma_f32_16x16x32_bf16 a[176:179], v034 v035 v036 v037, v038 v039 v040 v041, a[176:179]
	v_mfma_f32_16x16x32_bf16 a[172:175], v034 v035 v036 v037, v046 v047 v048 v049, a[172:175]
	ds_read_b128 v050 v051 v052 v053, v067 offset:20480
	v_mfma_f32_16x16x32_bf16 a[168:171], v034 v035 v036 v037, v054 v055 v056 v057, a[168:171]
	v_mfma_f32_16x16x32_bf16 a[164:167], v034 v035 v036 v037, v058 v059 v060 v061, a[164:167]
	v_mfma_f32_16x16x32_bf16 a[160:163], v034 v035 v036 v037, v062 v063 v064 v065, a[160:163]
	v_mfma_f32_16x16x32_bf16 a[156:159], v026 v027 v028 v029, v018 v019 v020 v021, a[156:159]
	ds_read_b128 v042 v043 v044 v045, v067 offset:21504
	v_mfma_f32_16x16x32_bf16 a[152:155], v026 v027 v028 v029, v022 v023 v024 v025, a[152:155]
	v_mfma_f32_16x16x32_bf16 a[148:151], v026 v027 v028 v029, v030 v031 v032 v033, a[148:151]
	v_mfma_f32_16x16x32_bf16 a[144:147], v026 v027 v028 v029, v038 v039 v040 v041, a[144:147]
	v_mfma_f32_16x16x32_bf16 a[140:143], v026 v027 v028 v029, v046 v047 v048 v049, a[140:143]
	ds_read_b128 v034 v035 v036 v037, v067 offset:22528
	v_add3_u32 v067, v099, s4, v096
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, -1
	s_and_b32 s1, s1, 0xffff
	v_mfma_f32_16x16x32_bf16 a[136:139], v026 v027 v028 v029, v054 v055 v056 v057, a[136:139]
	v_mfma_f32_16x16x32_bf16 a[132:135], v026 v027 v028 v029, v058 v059 v060 v061, a[132:135]
	v_mfma_f32_16x16x32_bf16 a[128:131], v026 v027 v028 v029, v062 v063 v064 v065, a[128:131]
	v_mfma_f32_16x16x32_bf16 a[124:127], v014 v015 v016 v017, v018 v019 v020 v021, a[124:127]
	ds_read_b128 v026 v027 v028 v029, v067 offset:16384
	v_mfma_f32_16x16x32_bf16 a[120:123], v014 v015 v016 v017, v022 v023 v024 v025, a[120:123]
	v_mfma_f32_16x16x32_bf16 a[116:119], v014 v015 v016 v017, v030 v031 v032 v033, a[116:119]
	v_mfma_f32_16x16x32_bf16 a[112:115], v014 v015 v016 v017, v038 v039 v040 v041, a[112:115]
	v_mfma_f32_16x16x32_bf16 a[108:111], v014 v015 v016 v017, v046 v047 v048 v049, a[108:111]
	ds_read_b128 v088 v089 v090 v091, v067 offset:17408
	v_mfma_f32_16x16x32_bf16 a[104:107], v014 v015 v016 v017, v054 v055 v056 v057, a[104:107]
	v_mfma_f32_16x16x32_bf16 a[100:103], v014 v015 v016 v017, v058 v059 v060 v061, a[100:103]
	v_mfma_f32_16x16x32_bf16 a[96:99], v014 v015 v016 v017, v062 v063 v064 v065, a[96:99]
	v_mfma_f32_16x16x32_bf16 a[92:95], v010 v011 v012 v013, v018 v019 v020 v021, a[92:95]
	ds_read_b128 v014 v015 v016 v017, v067 offset:18432
	v_mfma_f32_16x16x32_bf16 a[88:91], v010 v011 v012 v013, v022 v023 v024 v025, a[88:91]
	v_mfma_f32_16x16x32_bf16 a[84:87], v010 v011 v012 v013, v030 v031 v032 v033, a[84:87]
	v_mfma_f32_16x16x32_bf16 a[80:83], v010 v011 v012 v013, v038 v039 v040 v041, a[80:83]
	v_mfma_f32_16x16x32_bf16 a[76:79], v010 v011 v012 v013, v046 v047 v048 v049, a[76:79]
	ds_read_b128 v096 v097 v098 v099, v067 offset:19456
	v_mfma_f32_16x16x32_bf16 a[72:75], v010 v011 v012 v013, v054 v055 v056 v057, a[72:75]
	v_mfma_f32_16x16x32_bf16 a[68:71], v010 v011 v012 v013, v058 v059 v060 v061, a[68:71]
	v_mfma_f32_16x16x32_bf16 a[64:67], v010 v011 v012 v013, v062 v063 v064 v065, a[64:67]
	v_mfma_f32_16x16x32_bf16 a[60:63], v006 v007 v008 v009, v018 v019 v020 v021, a[60:63]
	ds_read_b128 v010 v011 v012 v013, v067 offset:20480
	v_mfma_f32_16x16x32_bf16 a[56:59], v006 v007 v008 v009, v022 v023 v024 v025, a[56:59]
	v_mfma_f32_16x16x32_bf16 a[52:55], v006 v007 v008 v009, v030 v031 v032 v033, a[52:55]
	v_mfma_f32_16x16x32_bf16 a[48:51], v006 v007 v008 v009, v038 v039 v040 v041, a[48:51]
	v_mfma_f32_16x16x32_bf16 a[44:47], v006 v007 v008 v009, v046 v047 v048 v049, a[44:47]
	ds_read_b128 v100 v101 v102 v103, v067 offset:21504
	v_mfma_f32_16x16x32_bf16 a[40:43], v006 v007 v008 v009, v054 v055 v056 v057, a[40:43]
	v_mfma_f32_16x16x32_bf16 a[36:39], v006 v007 v008 v009, v058 v059 v060 v061, a[36:39]
	v_mfma_f32_16x16x32_bf16 a[32:35], v006 v007 v008 v009, v062 v063 v064 v065, a[32:35]
	v_mfma_f32_16x16x32_bf16 a[28:31], v002 v003 v004 v005, v018 v019 v020 v021, a[28:31]
	ds_read_b128 v006 v007 v008 v009, v067 offset:22528
	v_mfma_f32_16x16x32_bf16 a[24:27], v002 v003 v004 v005, v022 v023 v024 v025, a[24:27]
	v_mfma_f32_16x16x32_bf16 a[20:23], v002 v003 v004 v005, v030 v031 v032 v033, a[20:23]
	v_mfma_f32_16x16x32_bf16 a[16:19], v002 v003 v004 v005, v038 v039 v040 v041, a[16:19]
	v_mfma_f32_16x16x32_bf16 a[12:15], v002 v003 v004 v005, v046 v047 v048 v049, a[12:15]
	ds_read_b128 v018 v019 v020 v021, v067 offset:23552
	v_mfma_f32_16x16x32_bf16 a[8:11], v002 v003 v004 v005, v054 v055 v056 v057, a[8:11]
	v_mfma_f32_16x16x32_bf16 a[4:7], v002 v003 v004 v005, v058 v059 v060 v061, a[4:7]
	v_mfma_f32_16x16x32_bf16 a[0:3], v002 v003 v004 v005, v062 v063 v064 v065, a[0:3]
	v_or_b32_e32 v066, 0x70, v001
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_bf16 a[252:255], v068 v069 v070 v071, v026 v027 v028 v029, a[252:255]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 a[248:251], v068 v069 v070 v071, v088 v089 v090 v091, a[248:251]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 a[244:247], v068 v069 v070 v071, v014 v015 v016 v017, a[244:247]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 a[240:243], v068 v069 v070 v071, v096 v097 v098 v099, a[240:243]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_bf16 a[236:239], v068 v069 v070 v071, v010 v011 v012 v013, a[236:239]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 a[232:235], v068 v069 v070 v071, v100 v101 v102 v103, a[232:235]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 a[228:231], v068 v069 v070 v071, v006 v007 v008 v009, a[228:231]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 a[224:227], v068 v069 v070 v071, v018 v019 v020 v021, a[224:227]
	v_mfma_f32_16x16x32_bf16 a[220:223], v072 v073 v074 v075, v026 v027 v028 v029, a[220:223]
	v_mfma_f32_16x16x32_bf16 a[216:219], v072 v073 v074 v075, v088 v089 v090 v091, a[216:219]
	v_mfma_f32_16x16x32_bf16 a[212:215], v072 v073 v074 v075, v014 v015 v016 v017, a[212:215]
	v_mfma_f32_16x16x32_bf16 a[208:211], v072 v073 v074 v075, v096 v097 v098 v099, a[208:211]
	v_mfma_f32_16x16x32_bf16 a[204:207], v072 v073 v074 v075, v010 v011 v012 v013, a[204:207]
	v_mfma_f32_16x16x32_bf16 a[200:203], v072 v073 v074 v075, v100 v101 v102 v103, a[200:203]
	v_mfma_f32_16x16x32_bf16 a[196:199], v072 v073 v074 v075, v006 v007 v008 v009, a[196:199]
	v_mfma_f32_16x16x32_bf16 a[192:195], v072 v073 v074 v075, v018 v019 v020 v021, a[192:195]
	v_mfma_f32_16x16x32_bf16 a[188:191], v076 v077 v078 v079, v026 v027 v028 v029, a[188:191]
	v_mfma_f32_16x16x32_bf16 a[184:187], v076 v077 v078 v079, v088 v089 v090 v091, a[184:187]
	v_mfma_f32_16x16x32_bf16 a[180:183], v076 v077 v078 v079, v014 v015 v016 v017, a[180:183]
	v_mfma_f32_16x16x32_bf16 a[176:179], v076 v077 v078 v079, v096 v097 v098 v099, a[176:179]
	v_mfma_f32_16x16x32_bf16 a[172:175], v076 v077 v078 v079, v010 v011 v012 v013, a[172:175]
	v_mfma_f32_16x16x32_bf16 a[168:171], v076 v077 v078 v079, v100 v101 v102 v103, a[168:171]
	v_mfma_f32_16x16x32_bf16 a[164:167], v076 v077 v078 v079, v006 v007 v008 v009, a[164:167]
	v_mfma_f32_16x16x32_bf16 a[160:163], v076 v077 v078 v079, v018 v019 v020 v021, a[160:163]
	v_mfma_f32_16x16x32_bf16 a[156:159], v080 v081 v082 v083, v026 v027 v028 v029, a[156:159]
	v_mfma_f32_16x16x32_bf16 a[152:155], v080 v081 v082 v083, v088 v089 v090 v091, a[152:155]
	v_mfma_f32_16x16x32_bf16 a[148:151], v080 v081 v082 v083, v014 v015 v016 v017, a[148:151]
	v_mfma_f32_16x16x32_bf16 a[144:147], v080 v081 v082 v083, v096 v097 v098 v099, a[144:147]
	v_mfma_f32_16x16x32_bf16 a[140:143], v080 v081 v082 v083, v010 v011 v012 v013, a[140:143]
	v_mfma_f32_16x16x32_bf16 a[136:139], v080 v081 v082 v083, v100 v101 v102 v103, a[136:139]
	v_mfma_f32_16x16x32_bf16 a[132:135], v080 v081 v082 v083, v006 v007 v008 v009, a[132:135]
	v_mfma_f32_16x16x32_bf16 a[128:131], v080 v081 v082 v083, v018 v019 v020 v021, a[128:131]
	v_mfma_f32_16x16x32_bf16 a[124:127], v084 v085 v086 v087, v026 v027 v028 v029, a[124:127]
	v_mfma_f32_16x16x32_bf16 a[120:123], v084 v085 v086 v087, v088 v089 v090 v091, a[120:123]
	v_mfma_f32_16x16x32_bf16 a[116:119], v084 v085 v086 v087, v014 v015 v016 v017, a[116:119]
	v_mfma_f32_16x16x32_bf16 a[112:115], v084 v085 v086 v087, v096 v097 v098 v099, a[112:115]
	v_mfma_f32_16x16x32_bf16 a[108:111], v084 v085 v086 v087, v010 v011 v012 v013, a[108:111]
	v_mfma_f32_16x16x32_bf16 a[104:107], v084 v085 v086 v087, v100 v101 v102 v103, a[104:107]
	v_mfma_f32_16x16x32_bf16 a[100:103], v084 v085 v086 v087, v006 v007 v008 v009, a[100:103]
	v_mfma_f32_16x16x32_bf16 a[96:99], v084 v085 v086 v087, v018 v019 v020 v021, a[96:99]
	v_mfma_f32_16x16x32_bf16 a[92:95], v050 v051 v052 v053, v026 v027 v028 v029, a[92:95]
	v_mfma_f32_16x16x32_bf16 a[88:91], v050 v051 v052 v053, v088 v089 v090 v091, a[88:91]
	v_mfma_f32_16x16x32_bf16 a[84:87], v050 v051 v052 v053, v014 v015 v016 v017, a[84:87]
	v_mfma_f32_16x16x32_bf16 a[80:83], v050 v051 v052 v053, v096 v097 v098 v099, a[80:83]
	v_mfma_f32_16x16x32_bf16 a[76:79], v050 v051 v052 v053, v010 v011 v012 v013, a[76:79]
	v_mfma_f32_16x16x32_bf16 a[72:75], v050 v051 v052 v053, v100 v101 v102 v103, a[72:75]
	v_mfma_f32_16x16x32_bf16 a[68:71], v050 v051 v052 v053, v006 v007 v008 v009, a[68:71]
	v_mfma_f32_16x16x32_bf16 a[64:67], v050 v051 v052 v053, v018 v019 v020 v021, a[64:67]
	v_mfma_f32_16x16x32_bf16 a[60:63], v042 v043 v044 v045, v026 v027 v028 v029, a[60:63]
	v_mfma_f32_16x16x32_bf16 a[56:59], v042 v043 v044 v045, v088 v089 v090 v091, a[56:59]
	v_mfma_f32_16x16x32_bf16 a[52:55], v042 v043 v044 v045, v014 v015 v016 v017, a[52:55]
	v_mfma_f32_16x16x32_bf16 a[48:51], v042 v043 v044 v045, v096 v097 v098 v099, a[48:51]
	v_mfma_f32_16x16x32_bf16 a[44:47], v042 v043 v044 v045, v010 v011 v012 v013, a[44:47]
	v_mfma_f32_16x16x32_bf16 a[40:43], v042 v043 v044 v045, v100 v101 v102 v103, a[40:43]
	v_mfma_f32_16x16x32_bf16 a[36:39], v042 v043 v044 v045, v006 v007 v008 v009, a[36:39]
	v_mfma_f32_16x16x32_bf16 a[32:35], v042 v043 v044 v045, v018 v019 v020 v021, a[32:35]
	v_mfma_f32_16x16x32_bf16 a[28:31], v034 v035 v036 v037, v026 v027 v028 v029, a[28:31]
	v_mfma_f32_16x16x32_bf16 a[24:27], v034 v035 v036 v037, v088 v089 v090 v091, a[24:27]
	v_mfma_f32_16x16x32_bf16 a[20:23], v034 v035 v036 v037, v014 v015 v016 v017, a[20:23]
	v_mfma_f32_16x16x32_bf16 a[16:19], v034 v035 v036 v037, v096 v097 v098 v099, a[16:19]
	v_mfma_f32_16x16x32_bf16 a[12:15], v034 v035 v036 v037, v010 v011 v012 v013, a[12:15]
	v_mfma_f32_16x16x32_bf16 a[8:11], v034 v035 v036 v037, v100 v101 v102 v103, a[8:11]
	v_mfma_f32_16x16x32_bf16 a[4:7], v034 v035 v036 v037, v006 v007 v008 v009, a[4:7]
	v_mfma_f32_16x16x32_bf16 a[0:3], v034 v035 v036 v037, v018 v019 v020 v021, a[0:3]
	v_lshlrev_b32_e32 v002, 2, v094
	v_accvgpr_read_b32 v003, a252
	v_and_or_b32 v004, v002, 12, v001
	v_cvt_pk_bf16_f32 v005, v003, s0
	v_lshlrev_b32_e32 v003, 1, v095
	v_lshl_or_b32 v004, v004, 9, v003
	s_barrier
	ds_write_b16 v004, v005
	v_accvgpr_read_b32 v005, a253
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:512
	v_accvgpr_read_b32 v005, a254
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:1024
	v_accvgpr_read_b32 v005, a255
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:1536
	v_accvgpr_read_b32 v005, a248
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:32
	v_accvgpr_read_b32 v005, a249
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:544
	v_accvgpr_read_b32 v005, a250
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:1056
	v_accvgpr_read_b32 v005, a251
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:1568
	v_accvgpr_read_b32 v005, a244
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:64
	v_accvgpr_read_b32 v005, a245
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:576
	v_accvgpr_read_b32 v005, a246
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:1088
	v_accvgpr_read_b32 v005, a247
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:1600
	v_accvgpr_read_b32 v005, a240
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:96
	v_accvgpr_read_b32 v005, a241
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:608
	v_accvgpr_read_b32 v005, a242
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:1120
	v_accvgpr_read_b32 v005, a243
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:1632
	v_accvgpr_read_b32 v005, a236
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:128
	v_accvgpr_read_b32 v005, a237
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:640
	v_accvgpr_read_b32 v005, a238
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:1152
	v_accvgpr_read_b32 v005, a239
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:1664
	v_accvgpr_read_b32 v005, a232
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:160
	v_accvgpr_read_b32 v005, a233
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:672
	v_accvgpr_read_b32 v005, a234
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:1184
	v_accvgpr_read_b32 v005, a235
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:1696
	v_accvgpr_read_b32 v005, a228
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:192
	v_accvgpr_read_b32 v005, a229
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:704
	v_accvgpr_read_b32 v005, a230
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:1216
	v_accvgpr_read_b32 v005, a231
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:1728
	v_accvgpr_read_b32 v005, a224
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:224
	v_accvgpr_read_b32 v005, a225
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:736
	v_accvgpr_read_b32 v005, a226
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:1248
	v_accvgpr_read_b32 v005, a227
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:1760
	v_accvgpr_read_b32 v005, a220
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:8192
	v_accvgpr_read_b32 v005, a221
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:8704
	v_accvgpr_read_b32 v005, a222
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:9216
	v_accvgpr_read_b32 v005, a223
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:9728
	v_accvgpr_read_b32 v005, a216
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:8224
	v_accvgpr_read_b32 v005, a217
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:8736
	v_accvgpr_read_b32 v005, a218
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:9248
	v_accvgpr_read_b32 v005, a219
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:9760
	v_accvgpr_read_b32 v005, a212
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:8256
	v_accvgpr_read_b32 v005, a213
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:8768
	v_accvgpr_read_b32 v005, a214
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:9280
	v_accvgpr_read_b32 v005, a215
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:9792
	v_accvgpr_read_b32 v005, a208
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:8288
	v_accvgpr_read_b32 v005, a209
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:8800
	v_accvgpr_read_b32 v005, a210
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:9312
	v_accvgpr_read_b32 v005, a211
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:9824
	v_accvgpr_read_b32 v005, a204
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:8320
	v_accvgpr_read_b32 v005, a205
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:8832
	v_accvgpr_read_b32 v005, a206
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:9344
	v_accvgpr_read_b32 v005, a207
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:9856
	v_accvgpr_read_b32 v005, a200
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:8352
	v_accvgpr_read_b32 v005, a201
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:8864
	v_accvgpr_read_b32 v005, a202
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:9376
	v_accvgpr_read_b32 v005, a203
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:9888
	v_accvgpr_read_b32 v005, a196
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:8384
	v_accvgpr_read_b32 v005, a197
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:8896
	v_accvgpr_read_b32 v005, a198
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:9408
	v_accvgpr_read_b32 v005, a199
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:9920
	v_accvgpr_read_b32 v005, a192
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:8416
	v_accvgpr_read_b32 v005, a193
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:8928
	v_accvgpr_read_b32 v005, a194
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:9440
	v_accvgpr_read_b32 v005, a195
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:9952
	v_accvgpr_read_b32 v005, a188
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:16384
	v_accvgpr_read_b32 v005, a189
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:16896
	v_accvgpr_read_b32 v005, a190
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:17408
	v_accvgpr_read_b32 v005, a191
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:17920
	v_accvgpr_read_b32 v005, a184
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:16416
	v_accvgpr_read_b32 v005, a185
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:16928
	v_accvgpr_read_b32 v005, a186
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:17440
	v_accvgpr_read_b32 v005, a187
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:17952
	v_accvgpr_read_b32 v005, a180
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:16448
	v_accvgpr_read_b32 v005, a181
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:16960
	v_accvgpr_read_b32 v005, a182
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:17472
	v_accvgpr_read_b32 v005, a183
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:17984
	v_accvgpr_read_b32 v005, a176
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:16480
	v_accvgpr_read_b32 v005, a177
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:16992
	v_accvgpr_read_b32 v005, a178
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:17504
	v_accvgpr_read_b32 v005, a179
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:18016
	v_accvgpr_read_b32 v005, a172
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:16512
	v_accvgpr_read_b32 v005, a173
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:17024
	v_accvgpr_read_b32 v005, a174
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:17536
	v_accvgpr_read_b32 v005, a175
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:18048
	v_accvgpr_read_b32 v005, a168
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:16544
	v_accvgpr_read_b32 v005, a169
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:17056
	v_accvgpr_read_b32 v005, a170
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:17568
	v_accvgpr_read_b32 v005, a171
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:18080
	v_accvgpr_read_b32 v005, a164
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:16576
	v_accvgpr_read_b32 v005, a165
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:17088
	v_accvgpr_read_b32 v005, a166
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:17600
	v_accvgpr_read_b32 v005, a167
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:18112
	v_accvgpr_read_b32 v005, a160
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:16608
	v_accvgpr_read_b32 v005, a161
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:17120
	v_accvgpr_read_b32 v005, a162
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:17632
	v_accvgpr_read_b32 v005, a163
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v004, v005 offset:18144
	v_or3_b32 v001, v001, 48, v002
	v_accvgpr_read_b32 v005, a156
	v_cvt_pk_bf16_f32 v005, v005, s0
	v_lshl_or_b32 v001, v001, 9, v003
	ds_write_b16 v001, v005
	v_accvgpr_read_b32 v005, a157
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:512
	v_accvgpr_read_b32 v005, a158
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:1024
	v_accvgpr_read_b32 v005, a159
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:1536
	v_accvgpr_read_b32 v005, a152
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:32
	v_accvgpr_read_b32 v005, a153
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:544
	v_accvgpr_read_b32 v005, a154
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:1056
	v_accvgpr_read_b32 v005, a155
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:1568
	v_accvgpr_read_b32 v005, a148
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:64
	v_accvgpr_read_b32 v005, a149
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:576
	v_accvgpr_read_b32 v005, a150
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:1088
	v_accvgpr_read_b32 v005, a151
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:1600
	v_accvgpr_read_b32 v005, a144
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:96
	v_accvgpr_read_b32 v005, a145
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:608
	v_accvgpr_read_b32 v005, a146
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:1120
	v_accvgpr_read_b32 v005, a147
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:1632
	v_accvgpr_read_b32 v005, a140
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:128
	v_accvgpr_read_b32 v005, a141
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:640
	v_accvgpr_read_b32 v005, a142
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:1152
	v_accvgpr_read_b32 v005, a143
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:1664
	v_accvgpr_read_b32 v005, a136
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:160
	v_accvgpr_read_b32 v005, a137
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:672
	v_accvgpr_read_b32 v005, a138
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:1184
	v_accvgpr_read_b32 v005, a139
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:1696
	v_accvgpr_read_b32 v005, a132
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:192
	v_accvgpr_read_b32 v005, a133
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:704
	v_accvgpr_read_b32 v005, a134
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:1216
	v_accvgpr_read_b32 v005, a135
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:1728
	v_accvgpr_read_b32 v005, a128
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:224
	v_accvgpr_read_b32 v005, a129
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:736
	v_accvgpr_read_b32 v005, a130
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:1248
	v_accvgpr_read_b32 v005, a131
	v_cvt_pk_bf16_f32 v005, v005, s0
	ds_write_b16 v001, v005 offset:1760
	v_accvgpr_read_b32 v001, a124
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:32768
	v_accvgpr_read_b32 v001, a125
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:33280
	v_accvgpr_read_b32 v001, a126
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:33792
	v_accvgpr_read_b32 v001, a127
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:34304
	v_accvgpr_read_b32 v001, a120
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:32800
	v_accvgpr_read_b32 v001, a121
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:33312
	v_accvgpr_read_b32 v001, a122
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:33824
	v_accvgpr_read_b32 v001, a123
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:34336
	v_accvgpr_read_b32 v001, a116
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:32832
	v_accvgpr_read_b32 v001, a117
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:33344
	v_accvgpr_read_b32 v001, a118
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:33856
	v_accvgpr_read_b32 v001, a119
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:34368
	v_accvgpr_read_b32 v001, a112
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:32864
	v_accvgpr_read_b32 v001, a113
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:33376
	v_accvgpr_read_b32 v001, a114
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:33888
	v_accvgpr_read_b32 v001, a115
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:34400
	v_accvgpr_read_b32 v001, a108
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:32896
	v_accvgpr_read_b32 v001, a109
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:33408
	v_accvgpr_read_b32 v001, a110
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:33920
	v_accvgpr_read_b32 v001, a111
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:34432
	v_accvgpr_read_b32 v001, a104
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:32928
	v_accvgpr_read_b32 v001, a105
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:33440
	v_accvgpr_read_b32 v001, a106
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:33952
	v_accvgpr_read_b32 v001, a107
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:34464
	v_accvgpr_read_b32 v001, a100
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:32960
	v_accvgpr_read_b32 v001, a101
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:33472
	v_accvgpr_read_b32 v001, a102
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:33984
	v_accvgpr_read_b32 v001, a103
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:34496
	v_accvgpr_read_b32 v001, a96
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:32992
	v_accvgpr_read_b32 v001, a97
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:33504
	v_accvgpr_read_b32 v001, a98
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:34016
	v_accvgpr_read_b32 v001, a99
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:34528
	v_accvgpr_read_b32 v001, a92
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:40960
	v_accvgpr_read_b32 v001, a93
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:41472
	v_accvgpr_read_b32 v001, a94
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:41984
	v_accvgpr_read_b32 v001, a95
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:42496
	v_accvgpr_read_b32 v001, a88
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:40992
	v_accvgpr_read_b32 v001, a89
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:41504
	v_accvgpr_read_b32 v001, a90
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:42016
	v_accvgpr_read_b32 v001, a91
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:42528
	v_accvgpr_read_b32 v001, a84
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:41024
	v_accvgpr_read_b32 v001, a85
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:41536
	v_accvgpr_read_b32 v001, a86
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:42048
	v_accvgpr_read_b32 v001, a87
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:42560
	v_accvgpr_read_b32 v001, a80
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:41056
	v_accvgpr_read_b32 v001, a81
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:41568
	v_accvgpr_read_b32 v001, a82
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:42080
	v_accvgpr_read_b32 v001, a83
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:42592
	v_accvgpr_read_b32 v001, a76
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:41088
	v_accvgpr_read_b32 v001, a77
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:41600
	v_accvgpr_read_b32 v001, a78
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:42112
	v_accvgpr_read_b32 v001, a79
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:42624
	v_accvgpr_read_b32 v001, a72
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:41120
	v_accvgpr_read_b32 v001, a73
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:41632
	v_accvgpr_read_b32 v001, a74
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:42144
	v_accvgpr_read_b32 v001, a75
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:42656
	v_accvgpr_read_b32 v001, a68
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:41152
	v_accvgpr_read_b32 v001, a69
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:41664
	v_accvgpr_read_b32 v001, a70
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:42176
	v_accvgpr_read_b32 v001, a71
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:42688
	v_accvgpr_read_b32 v001, a64
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:41184
	v_accvgpr_read_b32 v001, a65
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:41696
	v_accvgpr_read_b32 v001, a66
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:42208
	v_accvgpr_read_b32 v001, a67
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:42720
	v_accvgpr_read_b32 v001, a60
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:49152
	v_accvgpr_read_b32 v001, a61
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:49664
	v_accvgpr_read_b32 v001, a62
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:50176
	v_accvgpr_read_b32 v001, a63
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:50688
	v_accvgpr_read_b32 v001, a56
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:49184
	v_accvgpr_read_b32 v001, a57
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:49696
	v_accvgpr_read_b32 v001, a58
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:50208
	v_accvgpr_read_b32 v001, a59
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:50720
	v_accvgpr_read_b32 v001, a52
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:49216
	v_accvgpr_read_b32 v001, a53
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:49728
	v_accvgpr_read_b32 v001, a54
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:50240
	v_accvgpr_read_b32 v001, a55
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:50752
	v_accvgpr_read_b32 v001, a48
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:49248
	v_accvgpr_read_b32 v001, a49
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:49760
	v_accvgpr_read_b32 v001, a50
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:50272
	v_accvgpr_read_b32 v001, a51
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:50784
	v_accvgpr_read_b32 v001, a44
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:49280
	v_accvgpr_read_b32 v001, a45
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:49792
	v_accvgpr_read_b32 v001, a46
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:50304
	v_accvgpr_read_b32 v001, a47
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:50816
	v_accvgpr_read_b32 v001, a40
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:49312
	v_accvgpr_read_b32 v001, a41
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:49824
	v_accvgpr_read_b32 v001, a42
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:50336
	v_accvgpr_read_b32 v001, a43
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:50848
	v_accvgpr_read_b32 v001, a36
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:49344
	v_accvgpr_read_b32 v001, a37
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:49856
	v_accvgpr_read_b32 v001, a38
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:50368
	v_accvgpr_read_b32 v001, a39
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:50880
	v_accvgpr_read_b32 v001, a32
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:49376
	v_accvgpr_read_b32 v001, a33
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:49888
	v_accvgpr_read_b32 v001, a34
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:50400
	v_accvgpr_read_b32 v001, a35
	v_cvt_pk_bf16_f32 v001, v001, s0
	ds_write_b16 v004, v001 offset:50912
	v_or_b32_e32 v001, v002, v066
	v_accvgpr_read_b32 v002, a28
	v_cvt_pk_bf16_f32 v002, v002, s0
	v_lshl_or_b32 v001, v001, 9, v003
	ds_write_b16 v001, v002
	v_accvgpr_read_b32 v002, a29
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:512
	v_accvgpr_read_b32 v002, a30
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:1024
	v_accvgpr_read_b32 v002, a31
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:1536
	v_accvgpr_read_b32 v002, a24
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:32
	v_accvgpr_read_b32 v002, a25
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:544
	v_accvgpr_read_b32 v002, a26
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:1056
	v_accvgpr_read_b32 v002, a27
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:1568
	v_accvgpr_read_b32 v002, a20
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:64
	v_accvgpr_read_b32 v002, a21
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:576
	v_accvgpr_read_b32 v002, a22
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:1088
	v_accvgpr_read_b32 v002, a23
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:1600
	v_accvgpr_read_b32 v002, a16
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:96
	v_accvgpr_read_b32 v002, a17
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:608
	v_accvgpr_read_b32 v002, a18
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:1120
	v_accvgpr_read_b32 v002, a19
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:1632
	v_accvgpr_read_b32 v002, a12
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:128
	v_accvgpr_read_b32 v002, a13
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:640
	v_accvgpr_read_b32 v002, a14
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:1152
	v_accvgpr_read_b32 v002, a15
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:1664
	v_accvgpr_read_b32 v002, a8
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:160
	v_accvgpr_read_b32 v002, a9
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:672
	v_accvgpr_read_b32 v002, a10
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:1184
	v_accvgpr_read_b32 v002, a11
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:1696
	v_accvgpr_read_b32 v002, a4
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:192
	v_accvgpr_read_b32 v002, a5
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:704
	v_accvgpr_read_b32 v002, a6
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:1216
	v_accvgpr_read_b32 v002, a7
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:1728
	v_accvgpr_read_b32 v002, a0
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:224
	v_accvgpr_read_b32 v002, a1
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:736
	v_accvgpr_read_b32 v002, a2
	v_cvt_pk_bf16_f32 v002, v002, s0
	ds_write_b16 v001, v002 offset:1248
	v_accvgpr_read_b32 v002, a3
	v_cvt_pk_bf16_f32 v002, v002, s0
	v_lshrrev_b32_e32 v004, 5, v000
	ds_write_b16 v001, v002 offset:1760
	v_or_b32_e32 v002, s20, v004
	v_mov_b32_e32 v003, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v002 v003
	v_lshlrev_b32_e32 v003, 3, v000
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_4:
	s_cbranch_execz .LBB0_4
	v_and_b32_e32 v000, 0xf8, v003
	v_lshlrev_b32_e32 v001, 1, v000
	v_lshl_or_b32 v001, v004, 9, v001
	ds_read_b128 v006 v007 v008 v009, v001
	v_or_b32_e32 v000, s18, v000
	v_lshlrev_b32_e32 v001, 14, v002
	v_lshl_add_u32 v000, v000, 1, v001
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_4:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 8, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_6:
	s_cbranch_execz .LBB0_6
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_6:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 16, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_8:
	s_cbranch_execz .LBB0_8
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_8:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 24, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_10:
	s_cbranch_execz .LBB0_10
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_10:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 32, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_12:
	s_cbranch_execz .LBB0_12
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_12:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 40, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_14:
	s_cbranch_execz .LBB0_14
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_14:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 48, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_16:
	s_cbranch_execz .LBB0_16
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_16:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 56, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_18:
	s_cbranch_execz .LBB0_18
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_18:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 64, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_20:
	s_cbranch_execz .LBB0_20
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_20:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0x48, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_22:
	s_cbranch_execz .LBB0_22
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_22:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0x50, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_24:
	s_cbranch_execz .LBB0_24
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_24:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0x58, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_26:
	s_cbranch_execz .LBB0_26
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_26:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0x60, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_28:
	s_cbranch_execz .LBB0_28
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_28:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0x68, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_30:
	s_cbranch_execz .LBB0_30
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_30:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0x70, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_32:
	s_cbranch_execz .LBB0_32
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_32:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0x78, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_34:
	s_cbranch_execz .LBB0_34
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_34:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0x80, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_36:
	s_cbranch_execz .LBB0_36
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_36:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0x88, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_38:
	s_cbranch_execz .LBB0_38
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_38:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0x90, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_40:
	s_cbranch_execz .LBB0_40
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_40:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0x98, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_42:
	s_cbranch_execz .LBB0_42
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_42:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0xa0, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_44:
	s_cbranch_execz .LBB0_44
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_44:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0xa8, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_46:
	s_cbranch_execz .LBB0_46
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_46:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0xb0, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_48:
	s_cbranch_execz .LBB0_48
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_48:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0xb8, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_50:
	s_cbranch_execz .LBB0_50
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_50:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0xc0, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_52:
	s_cbranch_execz .LBB0_52
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_52:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0xc8, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_54:
	s_cbranch_execz .LBB0_54
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_54:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0xd0, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_56:
	s_cbranch_execz .LBB0_56
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_56:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0xd8, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_58:
	s_cbranch_execz .LBB0_58
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_58:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0xe0, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_60:
	s_cbranch_execz .LBB0_60
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_60:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0xe8, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_62:
	s_cbranch_execz .LBB0_62
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_62:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0xf0, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_64:
	s_cbranch_execz .LBB0_64
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v005, 1, v001
	v_lshl_or_b32 v002, v002, 9, v005
	ds_read_b128 v006 v007 v008 v009, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v006 v007 v008 v009, v000, s[0:3], 0 offen
.LBB0_64:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v002, 0xf8, v004
	v_or_b32_e32 v000, s20, v002
	v_mov_b32_e32 v001, s19
	v_cmp_gt_u64_e32 vcc, s[12:13], v000 v001
	s_and_saveexec_b64 s[4:5], vcc
.JUMP.LBB0_66:
	s_cbranch_execz .LBB0_66
	v_and_b32_e32 v001, 0xf8, v003
	v_lshlrev_b32_e32 v003, 1, v001
	v_lshl_or_b32 v002, v002, 9, v003
	ds_read_b128 v002 v003 v004 v005, v002
	v_or_b32_e32 v001, s18, v001
	v_lshlrev_b32_e32 v000, 14, v000
	v_lshl_add_u32 v000, v001, 1, v000
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v002 v003 v004 v005, v000, s[0:3], 0 offen
.LBB0_66:
	s_endpgm
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
