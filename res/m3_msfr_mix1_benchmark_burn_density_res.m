
% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 23:42:11 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 27 00:11:29 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651027331123 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01575E+00  9.99603E-01  9.99671E-01  1.00412E+00  9.89119E-01  9.97966E-01  1.00042E+00  1.00650E+00  9.94914E-01  9.94336E-01  9.89090E-01  1.00831E+00  1.00136E+00  1.00269E+00  9.94336E-01  9.98226E-01  9.96080E-01  1.00751E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  3.47491E-02 0.00036  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.65251E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30898E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45357E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.73829E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48298E+01 0.00024  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47317E+01 0.00024  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.47921E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.85651E+00 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333910 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33393E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33393E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.69379E+02 ;
RUNNING_TIME              (idx, 1)        =  2.93058E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03138E+00  1.03138E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.78450E-01  1.78450E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.80959E+01  2.80959E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.55675E+00  1.51153E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.78328E+01  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 9.19200 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98483E+00 2.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.13625E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22664.88;
MEMSIZE                   (idx, 1)        = 22480.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  6.00000E-04 ;
URES_EMAX                 (idx, 1)        =  9.99999E-02 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 3 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  2.68608E+15 ;
TOT_DECAY_HEAT            (idx, 1)        =  2.11238E+03 ;
TOT_SF_RATE               (idx, 1)        =  3.05952E+00 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.68608E+15 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  2.11238E+03 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  0.00000E+00 ;
INHALATION_TOXICITY       (idx, 1)        =  2.58171E+10 ;
INGESTION_TOXICITY        (idx, 1)        =  1.37045E+08 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.58171E+10 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.37045E+08 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  0.00000E+00 ;
SR90_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
TE132_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
I131_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
I132_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
CS134_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
CS137_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  2.24223E+10 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  0.00000E+00 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.05858E+11 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.09440E+10 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.22837E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.15175E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.78064E-01 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.55832E+18 0.00417  1.66199E-02 0.00410 ];
U233_FISS                 (idx, [1:   4]) = [  9.21995E+19 0.00052  9.83380E-01 6.9E-05 ];
TH232_CAPT                (idx, [1:   4]) = [  9.27261E+19 0.00056  7.49019E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34012E+19 0.00130  1.08255E-01 0.00129 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001804 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15357E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001804 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5132838 5.13796E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3887500 3.89127E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 981466 9.82311E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001804 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.99303E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36456E+20 3.6E-06  2.36456E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37769E+19 1.2E-07  9.37769E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.23770E+20 0.00030  1.09245E+20 0.00024  1.45252E+19 0.00166 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.17547E+20 0.00017  2.03022E+20 0.00013  1.45252E+19 0.00166 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.40946E+20 0.00035  2.40946E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.32281E+22 0.00026  3.14524E+22 0.00024  1.77570E+21 0.00163 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.36688E+19 0.00131 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.41216E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.07898E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29423E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.17398E-01 0.07924 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.79807E-01 0.00756 ];
SIX_FF_P                  (idx, [1:   2]) = [  5.50671E-04 0.01383 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.10487E+04 0.02613 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01778E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99990E-01 1.1E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.86110E-01 0.06437 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  4.38469E-01 0.06437 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52148E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99671E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.81177E-01 0.00045  1.08671E-01 0.00044  3.48401E-04 0.00865 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.81411E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.81403E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.81411E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.08832E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03104E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02985E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.32635E-02 0.00130 ];
IMP_EALF                  (idx, [1:   2]) = [  1.32779E-02 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.49576E-01 0.00156 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.49596E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.26365E-03 0.00552  2.65336E-04 0.02138  8.12663E-04 0.01050  5.95256E-04 0.01256  1.26713E-03 0.00916  2.68825E-04 0.01887  5.44370E-05 0.04262 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.47769E-01 0.01510  1.24755E-02 2.9E-05  3.23382E-02 9.5E-05  1.06010E-01 0.00059  2.97341E-01 0.00024  1.23562E+00 0.00022  5.21180E+00 0.03693 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.21306E-03 0.00834  2.73651E-04 0.03004  8.02851E-04 0.01525  5.86674E-04 0.01867  1.23967E-03 0.01317  2.59288E-04 0.02877  5.09191E-05 0.06286 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.43480E-01 0.02244  1.24752E-02 4.0E-05  3.23372E-02 0.00013  1.06034E-01 0.00084  2.97350E-01 0.00037  1.23563E+00 0.00029  6.24745E+00 0.02979 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.86337E-07 0.00118  8.86048E-07 0.00119  9.72402E-07 0.01762 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.69604E-07 0.00110  8.69320E-07 0.00111  9.54237E-07 0.01768 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.20768E-03 0.00892  2.66251E-04 0.03264  7.95681E-04 0.01791  5.92023E-04 0.02139  1.24099E-03 0.01476  2.57096E-04 0.03246  5.56440E-05 0.06297 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.44107E-01 0.02288  1.24753E-02 5.2E-05  3.23340E-02 0.00015  1.06007E-01 0.00096  2.97410E-01 0.00044  1.23625E+00 0.00036  5.82627E+00 0.04137 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.89230E-07 0.01967  7.88956E-07 0.01967  8.54093E-07 0.05940 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.74549E-07 0.01966  7.74280E-07 0.01966  8.38189E-07 0.05938 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.87615E-03 0.04224  2.74084E-04 0.12746  6.79970E-04 0.08373  5.72000E-04 0.08734  1.09188E-03 0.06349  2.02417E-04 0.12274  5.57998E-05 0.32322 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.51930E-01 0.11664  1.24750E-02 0.00015  3.23398E-02 0.00056  1.06075E-01 0.00327  2.97761E-01 0.00166  1.23540E+00 0.00113  6.79586E+00 0.13034 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.89135E-03 0.04175  2.67367E-04 0.12484  6.97149E-04 0.08285  5.73810E-04 0.08651  1.10657E-03 0.06239  1.95213E-04 0.11984  5.12440E-05 0.32460 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.42767E-01 0.11476  1.24748E-02 0.00015  3.23401E-02 0.00056  1.06099E-01 0.00328  2.97739E-01 0.00164  1.23534E+00 0.00113  6.81852E+00 0.12923 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.66470E+03 0.03773 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.80678E-07 0.00066 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.64055E-07 0.00052 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.23205E-03 0.00555 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.67056E+03 0.00561 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.25941E-09 0.00131 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.25107E-05 0.00377  7.25221E-05 0.00378  3.47902E-06 0.25747 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.93428E-05 0.01439  7.93243E-05 0.01444  4.30566E-06 0.36611 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  5.24747E-04 0.01357  5.24855E-04 0.01358  4.84075E-04 0.26417 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72125E+01 0.01155 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47317E+01 0.00024  4.03617E+01 0.00034 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 23:42:11 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 27 00:56:34 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651027331123 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00954E+00  1.00128E+00  1.00782E+00  9.99885E-01  9.92738E-01  9.97465E-01  9.97333E-01  1.00509E+00  9.94810E-01  9.91934E-01  9.98356E-01  9.96524E-01  9.94717E-01  1.00405E+00  1.00336E+00  9.98738E-01  1.00543E+00  1.00093E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 6.6E-10  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  3.46901E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.65310E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30979E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45413E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.73912E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48940E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47959E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.48926E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.85523E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333697 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33385E+04 0.00054 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33385E+04 0.00054 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.94729E+02 ;
RUNNING_TIME              (idx, 1)        =  7.43940E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03138E+00  1.03138E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  3.02650E-01  1.24200E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.30364E+01  4.49405E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.82000E-02  1.82000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.26983E+00  2.71302E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.17228E+01  4.38820E+02 ];
CPU_USAGE                 (idx, 1)        = 9.33851 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98918E+00 1.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.28705E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22664.88;
MEMSIZE                   (idx, 1)        = 22480.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 172 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  5.77830E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.60418E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.66358E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.48439E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.04621E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.26286E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46841E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.45800E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.65690E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.39581E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  4.79894E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06218E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17700E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44219E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35377E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.17374E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55147E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.60208E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.38114E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.39748E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51690E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.52785E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  7.31039E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.30587E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.26594E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.11353E-04 -2.39603E+25  2.15199E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.89042E-01 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.58033E+18 0.00390  1.68481E-02 0.00381 ];
U233_FISS                 (idx, [1:   4]) = [  9.21823E+19 0.00052  9.82830E-01 6.6E-05 ];
U235_FISS                 (idx, [1:   4]) = [  4.37657E+14 0.22892  4.67051E-06 0.22892 ];
TH232_CAPT                (idx, [1:   4]) = [  9.38313E+19 0.00051  7.44054E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34282E+19 0.00134  1.06482E-01 0.00129 ];
U235_CAPT                 (idx, [1:   4]) = [  2.50376E+13 1.00000  1.97800E-07 1.00000 ];
SM149_CAPT                (idx, [1:   4]) = [  1.40099E+16 0.03860  1.11062E-04 0.03860 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001553 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.17090E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001553 1.00117E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5172905 5.17840E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3847662 3.85141E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 980986 9.81891E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001553 1.00117E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.89990E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36450E+20 3.9E-06  2.36450E+20 3.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37766E+19 1.3E-07  9.37766E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.26068E+20 0.00030  1.11342E+20 0.00023  1.47268E+19 0.00165 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.19845E+20 0.00017  2.05118E+20 0.00013  1.47268E+19 0.00165 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.43529E+20 0.00036  2.43529E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.36126E+22 0.00025  3.18143E+22 0.00023  1.79820E+21 0.00154 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.39126E+19 0.00136 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.43757E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.09212E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29514E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29514E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.82287E-01 0.06591 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.89860E-01 0.00792 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.61663E-04 0.01526 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.06984E+04 0.02423 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01818E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99992E-01 8.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  5.99629E-01 0.05161 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.40880E-01 0.05161 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52142E+00 3.8E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.70976E-01 0.00042  1.07556E-01 0.00040  3.45535E-04 0.00891 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.71151E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.70969E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.71151E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.07691E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03184E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.03186E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.32528E-02 0.00129 ];
IMP_EALF                  (idx, [1:   2]) = [  1.32511E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.50632E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.50255E-01 0.00050 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.33708E-03 0.00533  2.72252E-04 0.02096  8.20425E-04 0.01069  6.15736E-04 0.01325  1.30383E-03 0.00834  2.74376E-04 0.01859  5.04609E-05 0.04819 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.43457E-01 0.01483  1.24343E-02 0.00334  3.23374E-02 9.6E-05  1.06041E-01 0.00057  2.97286E-01 0.00028  1.23519E+00 0.00022  5.26804E+00 0.04000 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.25056E-03 0.00804  2.56823E-04 0.02951  7.89105E-04 0.01598  6.08449E-04 0.01989  1.27830E-03 0.01350  2.69048E-04 0.02818  4.88406E-05 0.07403 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.42534E-01 0.02270  1.24760E-02 3.4E-05  3.23387E-02 0.00015  1.06092E-01 0.00084  2.97229E-01 0.00038  1.23528E+00 0.00030  6.58577E+00 0.02953 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.96143E-07 0.00121  8.95799E-07 0.00121  1.00600E-06 0.01743 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.70082E-07 0.00111  8.69747E-07 0.00111  9.76779E-07 0.01742 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.20520E-03 0.00896  2.61394E-04 0.03526  7.97171E-04 0.01857  5.95400E-04 0.02140  1.25199E-03 0.01466  2.52517E-04 0.03181  4.67339E-05 0.07847 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.37939E-01 0.02613  1.24766E-02 4.5E-05  3.23407E-02 0.00016  1.06027E-01 0.00100  2.97232E-01 0.00043  1.23474E+00 0.00044  6.49446E+00 0.04305 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.93686E-07 0.01958  7.93452E-07 0.01958  8.34996E-07 0.04780 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.70764E-07 0.01957  7.70537E-07 0.01957  8.10826E-07 0.04781 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.01059E-03 0.04352  2.51234E-04 0.13468  7.98457E-04 0.07500  5.85340E-04 0.08986  1.15309E-03 0.06178  1.77778E-04 0.14998  4.46867E-05 0.26540 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.26795E-01 0.09023  1.24758E-02 0.00013  3.23489E-02 0.00058  1.05900E-01 0.00319  2.96990E-01 0.00151  1.23411E+00 0.00134  6.17885E+00 0.11535 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.00747E-03 0.04279  2.52691E-04 0.13434  8.01951E-04 0.07351  5.75342E-04 0.08913  1.16004E-03 0.06000  1.73809E-04 0.15426  4.36390E-05 0.25258 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.21141E-01 0.08990  1.24757E-02 0.00013  3.23491E-02 0.00058  1.05903E-01 0.00318  2.96961E-01 0.00148  1.23411E+00 0.00134  6.17885E+00 0.11535 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.82785E+03 0.04011 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.89197E-07 0.00070 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.63347E-07 0.00058 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.22971E-03 0.00574 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.63322E+03 0.00586 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.17772E-09 0.00126 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.15763E-05 0.00410  7.15918E-05 0.00410  3.58813E-06 0.25381 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.74007E-05 0.01634  7.75169E-05 0.01636  2.20831E-06 0.34386 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.39635E-04 0.01490  4.39504E-04 0.01498  5.03723E-04 0.24442 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73203E+01 0.01146 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47959E+01 0.00027  4.04088E+01 0.00034 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 23:42:11 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 27 01:44:26 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651027331123 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00600E+00  1.00183E+00  1.00227E+00  1.00306E+00  9.99229E-01  1.00334E+00  9.97935E-01  9.92263E-01  9.95266E-01  1.00171E+00  1.00436E+00  9.97847E-01  9.97980E-01  9.97068E-01  1.00182E+00  9.98435E-01  1.00029E+00  9.99292E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  3.44466E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.65553E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31108E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45450E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.73522E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50506E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49530E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.51761E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.84547E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333968 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33377E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33377E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.14589E+03 ;
RUNNING_TIME              (idx, 1)        =  1.22254E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03138E+00  1.03138E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.33217E-01  1.30567E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.20738E+02  4.77017E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  3.71667E-02  1.89667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  7.18685E+00  2.91695E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.19379E+02  6.60460E+02 ];
CPU_USAGE                 (idx, 1)        = 9.37303 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98950E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.32225E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22664.88;
MEMSIZE                   (idx, 1)        = 22480.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 173 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.39973E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.65522E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.26425E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.87243E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.30971E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.49586E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49272E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  7.77145E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.40372E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.87559E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.03759E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.89564E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.59996E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.49267E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.34361E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33187E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54069E+18 ;
CS134_ACTIVITY            (idx, 1)        =  7.70328E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.17635E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.01763E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51657E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.36331E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.53704E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.39733E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.61906E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.92672E-04 -1.27528E+26  2.15302E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.06796E-01 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.59665E+18 0.00383  1.70331E-02 0.00378 ];
U233_FISS                 (idx, [1:   4]) = [  9.20033E+19 0.00050  9.81519E-01 6.7E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.17953E+16 0.04410  1.25802E-04 0.04408 ];
TH232_CAPT                (idx, [1:   4]) = [  9.52747E+19 0.00050  7.38375E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34238E+19 0.00147  1.04032E-01 0.00136 ];
U235_CAPT                 (idx, [1:   4]) = [  4.35520E+15 0.07478  3.37817E-05 0.07478 ];
SM149_CAPT                (idx, [1:   4]) = [  5.93213E+16 0.02173  4.59682E-04 0.02173 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001297 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15959E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001297 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5227398 5.23299E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3797678 3.80149E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 976221 9.77116E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001297 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.49012E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36435E+20 3.6E-06  2.36435E+20 3.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37760E+19 1.2E-07  9.37760E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.29019E+20 0.00029  1.14057E+20 0.00022  1.49627E+19 0.00162 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.22795E+20 0.00017  2.07833E+20 0.00012  1.49627E+19 0.00162 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.46578E+20 0.00035  2.46578E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.40936E+22 0.00024  3.22679E+22 0.00022  1.82567E+21 0.00157 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.40942E+19 0.00137 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.46890E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.10966E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29738E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29738E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.81685E-01 0.04686 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.21268E-01 0.00982 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.73837E-04 0.02326 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.10724E+03 0.02730 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.02293E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99994E-01 7.5E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.85950E-01 0.03428 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.09200E-01 0.03428 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52128E+00 3.5E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99673E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.58396E-01 0.00043  1.06147E-01 0.00041  3.49965E-04 0.00846 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.58771E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.58903E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.58771E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.06258E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04010E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04143E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.31445E-02 0.00140 ];
IMP_EALF                  (idx, [1:   2]) = [  1.31247E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.51351E-01 0.00160 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.50758E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.42008E-03 0.00538  2.87319E-04 0.01886  8.35277E-04 0.01129  6.17581E-04 0.01334  1.33951E-03 0.00832  2.88140E-04 0.01918  5.22594E-05 0.04369 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.42261E-01 0.01436  1.24758E-02 2.8E-05  3.23388E-02 9.5E-05  1.06138E-01 0.00056  2.97466E-01 0.00027  1.23536E+00 0.00024  5.16295E+00 0.03795 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.28202E-03 0.00821  2.80372E-04 0.02770  7.99667E-04 0.01648  5.87797E-04 0.01942  1.27543E-03 0.01285  2.85262E-04 0.02894  5.34938E-05 0.06245 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.46685E-01 0.01953  1.24760E-02 3.6E-05  3.23452E-02 0.00016  1.06203E-01 0.00086  2.97501E-01 0.00039  1.23502E+00 0.00036  6.15342E+00 0.02985 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.11974E-07 0.00115  9.11630E-07 0.00116  1.01899E-06 0.01735 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.73991E-07 0.00109  8.73661E-07 0.00110  9.76752E-07 0.01739 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.29159E-03 0.00891  2.93080E-04 0.03036  8.01366E-04 0.01789  5.86124E-04 0.02102  1.28730E-03 0.01429  2.77224E-04 0.03109  4.64904E-05 0.07750 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.33840E-01 0.02366  1.24757E-02 5.0E-05  3.23418E-02 0.00016  1.06135E-01 0.00101  2.97609E-01 0.00045  1.23624E+00 0.00036  6.07249E+00 0.04444 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.06078E-07 0.01966  8.05706E-07 0.01967  8.58260E-07 0.05493 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.72704E-07 0.01966  7.72347E-07 0.01966  8.22609E-07 0.05491 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.01807E-03 0.04308  2.34033E-04 0.13990  8.12489E-04 0.08041  4.94221E-04 0.08909  1.27303E-03 0.06557  1.60307E-04 0.13748  4.39892E-05 0.28265 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.13466E-01 0.10766  1.24773E-02 9.8E-05  3.23671E-02 0.00061  1.05966E-01 0.00326  2.96702E-01 0.00147  1.23465E+00 0.00127  6.24841E+00 0.13404 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.99321E-03 0.04261  2.38301E-04 0.13732  7.89675E-04 0.07915  4.95022E-04 0.08877  1.26167E-03 0.06534  1.66628E-04 0.13301  4.19137E-05 0.28461 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.17422E-01 0.10169  1.24772E-02 0.00010  3.23654E-02 0.00061  1.05923E-01 0.00319  2.96739E-01 0.00148  1.23466E+00 0.00127  6.31102E+00 0.13219 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.77518E+03 0.03877 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.05300E-07 0.00069 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.67582E-07 0.00050 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.30310E-03 0.00518 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.64918E+03 0.00524 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.96937E-09 0.00104 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.92283E-05 0.00614  6.91982E-05 0.00615  1.99516E-06 0.35809 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.94159E-05 0.02233  6.94409E-05 0.02235  1.83992E-06 0.50282 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.61570E-04 0.02284  2.61664E-04 0.02295  2.34987E-04 0.35278 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73706E+01 0.01215 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49530E+01 0.00025  4.05966E+01 0.00037 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 23:42:11 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 27 02:32:59 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651027331123 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00255E+00  9.99348E-01  1.00299E+00  1.00344E+00  9.97908E-01  9.99397E-01  9.94252E-01  1.00589E+00  9.96354E-01  1.00730E+00  9.96614E-01  9.96624E-01  1.00006E+00  9.98956E-01  9.92988E-01  1.00488E+00  9.91572E-01  1.00887E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  3.43288E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.65671E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.31219E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45518E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.72801E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51468E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50489E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.53323E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.84128E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333751 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33383E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33383E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.60374E+03 ;
RUNNING_TIME              (idx, 1)        =  1.70801E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03138E+00  1.03138E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.65567E-01  1.32350E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.69123E+02  4.83845E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  5.60500E-02  1.88833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.48167E-02  1.48167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.01220E+01  2.93513E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.67908E+02  6.94290E+02 ];
CPU_USAGE                 (idx, 1)        = 9.38951 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98957E+00 1.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.33913E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22664.88;
MEMSIZE                   (idx, 1)        = 22480.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 174 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.51842E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67122E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.27256E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91100E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33571E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.57571E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50586E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.83118E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.56612E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.03588E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.23604E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.79418E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.74249E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.48793E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.36076E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34501E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55834E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.39807E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.42541E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.09184E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.54593E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.29039E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.70064E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.46990E+15 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32019E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.15909E-03 -2.49406E+26  2.15424E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.20376E-01 0.00068 ];
TH232_FISS                (idx, [1:   4]) = [  1.61241E+18 0.00363  1.71770E-02 0.00359 ];
U233_FISS                 (idx, [1:   4]) = [  9.19779E+19 0.00048  9.79849E-01 7.0E-05 ];
U235_FISS                 (idx, [1:   4]) = [  4.82986E+16 0.02213  5.14454E-04 0.02210 ];
TH232_CAPT                (idx, [1:   4]) = [  9.63309E+19 0.00050  7.35262E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33654E+19 0.00132  1.02015E-01 0.00130 ];
U235_CAPT                 (idx, [1:   4]) = [  1.68519E+16 0.03626  1.28704E-04 0.03628 ];
SM149_CAPT                (idx, [1:   4]) = [  1.00982E+17 0.01582  7.70647E-04 0.01580 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001504 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14871E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001504 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5256317 5.26180E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3766300 3.76994E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 978887 9.79752E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001504 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.43424E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36431E+20 3.5E-06  2.36431E+20 3.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37752E+19 1.1E-07  9.37752E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.31109E+20 0.00029  1.15925E+20 0.00022  1.51845E+19 0.00170 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.24884E+20 0.00017  2.09700E+20 0.00012  1.51845E+19 0.00170 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.48997E+20 0.00033  2.48997E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.44549E+22 0.00024  3.26111E+22 0.00022  1.84378E+21 0.00170 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.43963E+19 0.00137 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.49281E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.12293E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29705E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29705E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.18154E-01 0.04780 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.24339E-01 0.01048 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.28088E-04 0.02141 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.66805E+03 0.02946 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.02029E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 8.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.97693E-01 0.03280 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.19507E-01 0.03280 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52125E+00 3.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.50526E-01 0.00042  1.05274E-01 0.00040  3.37291E-04 0.00916 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.49563E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.49565E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.49563E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.05270E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04277E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04336E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.31088E-02 0.00129 ];
IMP_EALF                  (idx, [1:   2]) = [  1.30995E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.52501E-01 0.00155 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.52125E-01 0.00044 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.42608E-03 0.00584  2.74625E-04 0.01971  8.60598E-04 0.01152  6.12664E-04 0.01392  1.33914E-03 0.00899  2.85949E-04 0.01812  5.31085E-05 0.04676 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.46866E-01 0.01577  1.24759E-02 2.9E-05  3.23366E-02 9.6E-05  1.06088E-01 0.00058  2.97589E-01 0.00027  1.23590E+00 0.00026  4.99173E+00 0.04011 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.25424E-03 0.00764  2.62749E-04 0.02969  8.22747E-04 0.01619  5.87446E-04 0.02015  1.26493E-03 0.01204  2.69687E-04 0.02821  4.66804E-05 0.06332 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.36282E-01 0.01912  1.24762E-02 3.7E-05  3.23319E-02 0.00013  1.06067E-01 0.00079  2.97552E-01 0.00040  1.23588E+00 0.00033  6.25433E+00 0.02963 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.21689E-07 0.00116  9.21335E-07 0.00116  1.03118E-06 0.01661 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.76042E-07 0.00108  8.75706E-07 0.00108  9.80235E-07 0.01667 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.18988E-03 0.00940  2.42225E-04 0.03423  8.03738E-04 0.01831  5.70618E-04 0.02229  1.27117E-03 0.01431  2.56478E-04 0.03139  4.56558E-05 0.08039 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.35626E-01 0.02492  1.24762E-02 4.7E-05  3.23383E-02 0.00017  1.06037E-01 0.00099  2.97603E-01 0.00048  1.23588E+00 0.00044  6.25028E+00 0.04387 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.13888E-07 0.01961  8.13553E-07 0.01961  9.61864E-07 0.06902 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.73751E-07 0.01961  7.73431E-07 0.01961  9.14787E-07 0.06917 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.57840E-03 0.04502  2.18921E-04 0.14370  6.10934E-04 0.08335  5.19054E-04 0.09832  9.86962E-04 0.06877  2.12027E-04 0.13384  3.04996E-05 0.30178 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.28136E-01 0.08969  1.24753E-02 0.00016  3.23535E-02 0.00063  1.06052E-01 0.00309  2.97601E-01 0.00175  1.23675E+00 0.00124  6.75662E+00 0.13247 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.58464E-03 0.04439  2.22091E-04 0.14268  6.08978E-04 0.08353  5.18562E-04 0.09531  9.82972E-04 0.06711  2.19596E-04 0.13268  3.24393E-05 0.29369 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.31768E-01 0.09161  1.24754E-02 0.00016  3.23533E-02 0.00064  1.06039E-01 0.00304  2.97614E-01 0.00172  1.23676E+00 0.00123  6.75662E+00 0.13247 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.18242E+03 0.04119 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.14695E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.69396E-07 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.13901E-03 0.00627 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.43265E+03 0.00637 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.91441E-09 0.00081 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.85593E-05 0.00646  6.85153E-05 0.00644  1.48509E-06 0.44515 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.36435E-05 0.02339  6.37361E-05 0.02334  6.44798E-07 0.57678 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.17668E-04 0.02111  2.17814E-04 0.02113  1.67168E-04 0.40613 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72809E+01 0.01124 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50489E+01 0.00026  4.07010E+01 0.00035 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 23:42:11 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 27 03:23:09 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651027331123 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00649E+00  1.00842E+00  1.00305E+00  1.00161E+00  9.95309E-01  9.95980E-01  1.00090E+00  1.00730E+00  1.00018E+00  1.00086E+00  9.98641E-01  9.97264E-01  9.99910E-01  9.96127E-01  9.92496E-01  9.93545E-01  9.97979E-01  1.00395E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  3.42591E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.65741E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30983E-01 0.00020  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45266E-01 0.00018  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.72631E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.53092E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.52114E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.57354E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.84333E+00 0.00048  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334071 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33395E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33395E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.07722E+03 ;
RUNNING_TIME              (idx, 1)        =  2.20973E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03138E+00  1.03138E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  7.01950E-01  1.36383E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.19125E+02  5.00024E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  7.42000E-02  1.81500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.48167E-02  1.48167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.31167E+01  2.99458E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.18019E+02  7.03410E+02 ];
CPU_USAGE                 (idx, 1)        = 9.40037 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98960E+00 1.7E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.35044E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22664.88;
MEMSIZE                   (idx, 1)        = 22480.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 175 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.54698E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66608E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.66855E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.93508E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35084E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.57982E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49887E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.00259E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.68727E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.32210E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.34444E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.69784E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.85271E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.93604E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.32417E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31696E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51973E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.34973E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.81399E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.09044E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51987E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.56681E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.75051E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.56327E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64039E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.35249E-03 -5.06197E+26  2.15681E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.45121E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.62805E+18 0.00376  1.73624E-02 0.00369 ];
U233_FISS                 (idx, [1:   4]) = [  9.15263E+19 0.00051  9.76124E-01 7.6E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.76441E+17 0.01258  1.88210E-03 0.01263 ];
TH232_CAPT                (idx, [1:   4]) = [  9.79734E+19 0.00051  7.31347E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33114E+19 0.00133  9.93678E-02 0.00130 ];
U235_CAPT                 (idx, [1:   4]) = [  6.12136E+16 0.02080  4.56867E-04 0.02077 ];
SM149_CAPT                (idx, [1:   4]) = [  1.41163E+17 0.01426  1.05375E-03 0.01426 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001863 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16912E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001863 1.00117E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5308388 5.31374E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3715647 3.71925E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 977828 9.78699E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001863 1.00117E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.63913E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36416E+20 3.8E-06  2.36416E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37729E+19 1.2E-07  9.37729E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.34096E+20 0.00031  1.18750E+20 0.00022  1.53456E+19 0.00189 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.27869E+20 0.00018  2.12523E+20 0.00012  1.53456E+19 0.00189 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.52109E+20 0.00034  2.52109E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.49813E+22 0.00025  3.31210E+22 0.00022  1.86030E+21 0.00193 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.46747E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.52543E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.14109E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.29337E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.29337E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.41868E-01 0.04988 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.24118E-01 0.01261 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.91760E-04 0.02576 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.08380E+03 0.02862 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.02133E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.30826E-01 0.03756 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.59412E-01 0.03756 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52115E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99680E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.37554E-01 0.00044  1.03848E-01 0.00044  3.36905E-04 0.00929 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.37233E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.37785E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.37233E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.03885E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04601E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04621E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.30665E-02 0.00132 ];
IMP_EALF                  (idx, [1:   2]) = [  1.30622E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.54276E-01 0.00148 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.54359E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.48739E-03 0.00569  2.88327E-04 0.01917  8.52624E-04 0.01149  6.40747E-04 0.01261  1.37235E-03 0.00928  2.82486E-04 0.01938  5.08494E-05 0.04489 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.38343E-01 0.01436  1.24760E-02 2.6E-05  3.23309E-02 9.1E-05  1.06231E-01 0.00061  2.97617E-01 0.00030  1.23709E+00 0.00030  5.22591E+00 0.03995 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.29493E-03 0.00838  2.61591E-04 0.02684  8.19609E-04 0.01672  5.99306E-04 0.01800  1.29947E-03 0.01367  2.67463E-04 0.02813  4.75002E-05 0.06695 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.37883E-01 0.02128  1.24762E-02 3.1E-05  3.23291E-02 0.00014  1.06285E-01 0.00091  2.97567E-01 0.00038  1.23740E+00 0.00037  6.47674E+00 0.03033 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.38153E-07 0.00116  9.37696E-07 0.00117  1.08119E-06 0.01636 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.79518E-07 0.00108  8.79089E-07 0.00108  1.01389E-06 0.01643 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.22873E-03 0.00950  2.55630E-04 0.03354  7.87290E-04 0.01907  6.02413E-04 0.02150  1.27294E-03 0.01382  2.61029E-04 0.03607  4.94302E-05 0.07040 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.41460E-01 0.02369  1.24763E-02 4.4E-05  3.23283E-02 0.00015  1.06287E-01 0.00106  2.97612E-01 0.00047  1.23733E+00 0.00058  6.21240E+00 0.04364 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.30480E-07 0.01958  8.30169E-07 0.01958  9.23674E-07 0.06011 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.78918E-07 0.01958  7.78627E-07 0.01958  8.66202E-07 0.06006 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.84864E-03 0.04300  2.38603E-04 0.13443  7.42697E-04 0.08232  5.78332E-04 0.09065  1.04954E-03 0.06951  1.92795E-04 0.13245  4.66687E-05 0.28278 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.48218E-01 0.11542  1.24757E-02 0.00014  3.23191E-02 0.00049  1.06500E-01 0.00372  2.98073E-01 0.00173  1.23866E+00 0.00104  5.84435E+00 0.13488 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.82968E-03 0.04210  2.36076E-04 0.13055  7.38249E-04 0.08006  5.69632E-04 0.08822  1.04953E-03 0.06753  1.91450E-04 0.12789  4.47481E-05 0.28271 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.40659E-01 0.11126  1.24758E-02 0.00014  3.23202E-02 0.00049  1.06487E-01 0.00370  2.98000E-01 0.00170  1.23874E+00 0.00105  5.84435E+00 0.13488 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.45435E+03 0.03877 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.32576E-07 0.00084 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.74287E-07 0.00070 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.18387E-03 0.00686 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.41406E+03 0.00678 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.89902E-09 0.00076 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.63863E-05 0.00805  6.64024E-05 0.00805  1.22295E-06 0.40860 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.01276E-05 0.02459  6.00686E-05 0.02461  1.61099E-06 0.51655 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.84623E-04 0.02510  1.84670E-04 0.02511  1.77621E-04 0.40614 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.69956E+01 0.01179 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.52114E+01 0.00028  4.08334E+01 0.00035 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 23:42:11 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 27 04:14:02 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651027331123 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00177E+00  1.00593E+00  1.00175E+00  9.98420E-01  9.97758E-01  9.98273E-01  9.99768E-01  1.00183E+00  1.00500E+00  9.93103E-01  9.97072E-01  9.94534E-01  9.99866E-01  1.00560E+00  9.97058E-01  9.97205E-01  1.00252E+00  1.00254E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  3.42097E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.65790E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30753E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.45024E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.72463E+00 0.00033  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.54192E+01 0.00024  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.53211E+01 0.00024  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.60359E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.84598E+00 0.00044  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333783 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33383E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33383E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.55035E+03 ;
RUNNING_TIME              (idx, 1)        =  2.71850E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03138E+00  1.03138E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  8.40117E-01  1.38167E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.69827E+02  5.07022E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  9.31167E-02  1.89167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.48167E-02  1.48167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.69169E+01  3.80022E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.68092E+02  7.19527E+02 ];
CPU_USAGE                 (idx, 1)        = 9.38148 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98876E+00 2.1E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.32951E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22664.88;
MEMSIZE                   (idx, 1)        = 22480.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 175 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.61680E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67607E+08 ;
TOT_SF_RATE               (idx, 1)        =  8.19339E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.97097E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37456E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61333E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50607E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.10169E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.79844E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.74745E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.50922E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.25409E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.94722E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.35096E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.33501E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.32487E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.53050E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.02072E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.17214E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.13106E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.55418E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.46994E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.85397E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.65422E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.96058E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -3.52790E-03 -7.59113E+26  2.15934E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.68143E-01 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.64072E+18 0.00402  1.75089E-02 0.00393 ];
U233_FISS                 (idx, [1:   4]) = [  9.10403E+19 0.00051  9.71598E-01 8.7E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.72887E+17 0.00846  3.97961E-03 0.00845 ];
TH232_CAPT                (idx, [1:   4]) = [  9.95136E+19 0.00053  7.28059E-01 0.00026 ];
U233_CAPT                 (idx, [1:   4]) = [  1.32734E+19 0.00127  9.71135E-02 0.00126 ];
U235_CAPT                 (idx, [1:   4]) = [  1.27730E+17 0.01438  9.34596E-04 0.01440 ];
SM149_CAPT                (idx, [1:   4]) = [  1.60825E+17 0.01309  1.17684E-03 0.01313 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001499 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15287E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001499 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5351701 5.35718E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3668910 3.67258E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 980888 9.81769E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001499 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.88127E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36400E+20 3.7E-06  2.36400E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37698E+19 1.2E-07  9.37698E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.36671E+20 0.00029  1.21111E+20 0.00021  1.55594E+19 0.00176 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.30440E+20 0.00017  2.14881E+20 0.00012  1.55594E+19 0.00176 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.55141E+20 0.00036  2.55141E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.54742E+22 0.00025  3.35904E+22 0.00023  1.88377E+21 0.00166 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.50496E+19 0.00137 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.55490E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.15759E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.28832E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.28832E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.12661E-01 0.04610 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.37052E-01 0.01189 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.94172E-04 0.02446 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.59585E+03 0.02884 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01826E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.80294E-01 0.03250 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.03650E-01 0.03250 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52107E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99686E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.25687E-01 0.00043  1.02545E-01 0.00043  3.28991E-04 0.00876 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.26360E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.26584E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.26360E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02718E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.04462E+00 0.00018 ];
IMP_ALF                   (idx, [1:   2]) = [  7.04222E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.30846E-02 0.00129 ];
IMP_EALF                  (idx, [1:   2]) = [  1.31144E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.56589E-01 0.00143 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.56933E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.49471E-03 0.00577  2.84802E-04 0.01992  8.51790E-04 0.01087  6.33299E-04 0.01428  1.36824E-03 0.00910  2.99194E-04 0.01856  5.73832E-05 0.04600 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.57459E-01 0.01565  1.24753E-02 2.9E-05  3.23290E-02 9.7E-05  1.06261E-01 0.00057  2.97754E-01 0.00028  1.23721E+00 0.00031  5.26549E+00 0.03928 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.22851E-03 0.00797  2.62991E-04 0.03011  8.00706E-04 0.01639  5.85286E-04 0.01982  1.25126E-03 0.01281  2.78289E-04 0.02707  4.99816E-05 0.06298 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.54054E-01 0.02230  1.24747E-02 4.3E-05  3.23240E-02 0.00014  1.06160E-01 0.00082  2.97734E-01 0.00041  1.23733E+00 0.00041  6.47361E+00 0.02912 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.49756E-07 0.00116  9.49462E-07 0.00116  1.04580E-06 0.01517 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.79133E-07 0.00109  8.78860E-07 0.00109  9.68136E-07 0.01517 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.18709E-03 0.00875  2.52849E-04 0.03540  7.76092E-04 0.01822  5.72557E-04 0.02149  1.26178E-03 0.01572  2.71249E-04 0.02895  5.25631E-05 0.07771 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.56549E-01 0.02512  1.24746E-02 5.7E-05  3.23300E-02 0.00018  1.06121E-01 0.00098  2.97645E-01 0.00050  1.23650E+00 0.00054  6.49271E+00 0.04094 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.39701E-07 0.01963  8.39439E-07 0.01963  8.87193E-07 0.05637 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.77467E-07 0.01962  7.77226E-07 0.01962  8.21443E-07 0.05637 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.96571E-03 0.04552  1.86619E-04 0.15639  6.76594E-04 0.07797  6.23537E-04 0.09250  1.19361E-03 0.06524  2.28624E-04 0.14180  5.67175E-05 0.31837 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.35334E-01 0.09558  1.24701E-02 0.00026  3.23410E-02 0.00056  1.06174E-01 0.00304  2.97608E-01 0.00156  1.23786E+00 0.00214  4.72894E+00 0.13372 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.97248E-03 0.04431  1.81009E-04 0.15466  6.89490E-04 0.07823  6.20174E-04 0.09023  1.19579E-03 0.06239  2.25125E-04 0.13825  6.08883E-05 0.30979 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.37036E-01 0.09527  1.24701E-02 0.00026  3.23398E-02 0.00055  1.06182E-01 0.00304  2.97634E-01 0.00154  1.23793E+00 0.00212  4.71743E+00 0.13415 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.55847E+03 0.04164 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.43851E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.73660E-07 0.00055 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.19224E-03 0.00615 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.38254E+03 0.00616 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.90681E-09 0.00072 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.66298E-05 0.00887  6.66141E-05 0.00888  1.06130E-06 0.51236 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.88103E-05 0.02650  5.88271E-05 0.02650  5.95931E-07 0.63677 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.85488E-04 0.02441  1.85739E-04 0.02437  1.20934E-04 0.50203 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71128E+01 0.01182 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.53211E+01 0.00024  4.08707E+01 0.00035 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 23:42:11 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 27 05:05:48 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651027331123 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01053E+00  1.00543E+00  1.00293E+00  1.00110E+00  9.94779E-01  1.00858E+00  9.98955E-01  9.92971E-01  1.00066E+00  9.93638E-01  9.99470E-01  9.97201E-01  1.00579E+00  9.97147E-01  9.90869E-01  9.96740E-01  9.98092E-01  1.00512E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 9.3E-10  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  3.41950E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.65805E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30660E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44927E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.72176E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.55467E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.54481E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.63137E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.85106E+00 0.00048  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333510 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33389E+04 0.00065 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33389E+04 0.00065 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.02654E+03 ;
RUNNING_TIME              (idx, 1)        =  3.23617E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03138E+00  1.03138E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  9.81833E-01  1.41717E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.21414E+02  5.15867E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.12683E-01  1.95667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.48167E-02  1.48167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.13742E+01  4.45723E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.19200E+02  7.26108E+02 ];
CPU_USAGE                 (idx, 1)        = 9.35223 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98983E+00 1.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.29671E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22664.88;
MEMSIZE                   (idx, 1)        = 22480.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 177 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.64950E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67519E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.25727E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.00051E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39236E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.61606E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50302E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.27783E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.93449E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.79111E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.67337E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.93976E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.06622E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.06876E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.30926E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30411E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.50252E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.83826E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.78526E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.13978E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.56846E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.24922E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.90671E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.77314E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.60097E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.96417E-03 -1.28334E+27  2.16458E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.96857E-01 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.65460E+18 0.00384  1.76517E-02 0.00375 ];
U233_FISS                 (idx, [1:   4]) = [  9.01487E+19 0.00057  9.61772E-01 9.5E-05 ];
U235_FISS                 (idx, [1:   4]) = [  9.03933E+17 0.00511  9.64314E-03 0.00504 ];
TH232_CAPT                (idx, [1:   4]) = [  1.01080E+20 0.00051  7.21536E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.31514E+19 0.00144  9.38777E-02 0.00137 ];
U235_CAPT                 (idx, [1:   4]) = [  3.16761E+17 0.00978  2.26106E-03 0.00975 ];
SM149_CAPT                (idx, [1:   4]) = [  1.77517E+17 0.01175  1.26751E-03 0.01181 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001667 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15047E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001667 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5401348 5.40675E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3614058 3.61754E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 986261 9.87218E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001667 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.19792E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36362E+20 3.7E-06  2.36362E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37618E+19 1.1E-07  9.37618E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.40105E+20 0.00029  1.24173E+20 0.00022  1.59312E+19 0.00163 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.33866E+20 0.00017  2.17935E+20 0.00013  1.59312E+19 0.00163 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59105E+20 0.00036  2.59105E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.61015E+22 0.00025  3.41792E+22 0.00023  1.92230E+21 0.00172 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.55798E+19 0.00131 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59446E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.17886E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.27805E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.27805E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.18773E-01 0.05081 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.24525E-01 0.01359 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.87586E-04 0.02419 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.45902E+03 0.02958 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01281E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.28770E-01 0.03607 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.56782E-01 0.03607 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52088E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99703E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.11700E-01 0.00045  1.00994E-01 0.00045  3.31140E-04 0.00951 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.12079E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.12261E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.12079E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  1.01197E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.03766E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.03575E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.31762E-02 0.00133 ];
IMP_EALF                  (idx, [1:   2]) = [  1.31996E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.61357E-01 0.00143 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.61588E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.63612E-03 0.00585  2.84179E-04 0.01946  8.82659E-04 0.01034  6.59956E-04 0.01395  1.43578E-03 0.00942  3.11048E-04 0.01891  6.25002E-05 0.04466 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.61158E-01 0.01551  1.24751E-02 2.9E-05  3.23196E-02 0.00011  1.06370E-01 0.00062  2.98118E-01 0.00026  1.24093E+00 0.00050  5.31355E+00 0.03559 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.28683E-03 0.00802  2.55654E-04 0.02897  8.09211E-04 0.01657  6.00002E-04 0.01832  1.28475E-03 0.01419  2.79509E-04 0.02827  5.77012E-05 0.06294 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.62847E-01 0.02176  1.24744E-02 4.5E-05  3.23150E-02 0.00015  1.06332E-01 0.00083  2.98075E-01 0.00040  1.24089E+00 0.00065  6.36762E+00 0.02810 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.68106E-07 0.00133  9.67696E-07 0.00132  1.10159E-06 0.02002 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.82554E-07 0.00120  8.82180E-07 0.00120  1.00417E-06 0.01996 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.26281E-03 0.00976  2.48272E-04 0.03546  8.05977E-04 0.01877  5.87114E-04 0.02261  1.28557E-03 0.01508  2.81223E-04 0.03019  5.46536E-05 0.07203 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.59025E-01 0.02625  1.24742E-02 6.8E-05  3.23231E-02 0.00019  1.06289E-01 0.00103  2.98162E-01 0.00046  1.24180E+00 0.00077  6.27269E+00 0.04147 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.53718E-07 0.01965  8.53568E-07 0.01965  9.14928E-07 0.05730 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.78329E-07 0.01964  7.78191E-07 0.01964  8.34612E-07 0.05740 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.88348E-03 0.04380  1.29600E-04 0.15443  7.26533E-04 0.08602  4.63426E-04 0.08670  1.24965E-03 0.06620  2.72710E-04 0.14826  4.15626E-05 0.28992 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.46656E-01 0.08314  1.24732E-02 0.00022  3.23085E-02 0.00067  1.06125E-01 0.00332  2.98076E-01 0.00171  1.23944E+00 0.00185  6.49064E+00 0.12315 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.89793E-03 0.04350  1.32521E-04 0.15477  7.38236E-04 0.08646  4.57548E-04 0.08431  1.24256E-03 0.06449  2.81036E-04 0.14547  4.60244E-05 0.29544 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.55337E-01 0.08870  1.24732E-02 0.00022  3.23082E-02 0.00066  1.06112E-01 0.00329  2.98037E-01 0.00171  1.23950E+00 0.00186  6.49064E+00 0.12315 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.38816E+03 0.03877 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.59570E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.74785E-07 0.00056 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.23963E-03 0.00602 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.37701E+03 0.00611 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.92457E-09 0.00077 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.71525E-05 0.00892  6.71816E-05 0.00891  1.86225E-06 0.33735 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.97005E-05 0.02602  5.97018E-05 0.02608  1.89016E-06 0.39927 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.78758E-04 0.02337  1.78512E-04 0.02340  2.43916E-04 0.32943 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.66849E+01 0.01146 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.54481E+01 0.00028  4.08699E+01 0.00035 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 23:42:11 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 27 05:58:16 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651027331123 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00169E+00  9.93259E-01  1.00207E+00  1.00770E+00  9.88810E-01  1.00274E+00  1.00612E+00  1.00097E+00  9.97988E-01  1.00119E+00  9.97679E-01  1.00174E+00  9.96812E-01  1.00364E+00  1.00279E+00  9.97174E-01  9.96880E-01  1.00076E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  3.42256E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.65774E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30330E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44612E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.72109E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.55726E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.54741E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.64838E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.85602E+00 0.00048  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333991 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33371E+04 0.00065 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33371E+04 0.00065 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.50979E+03 ;
RUNNING_TIME              (idx, 1)        =  3.76083E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03138E+00  1.03138E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.12635E+00  1.44517E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.73694E+02  5.22806E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.31350E-01  1.86667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.91333E-02  1.43167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.58200E+01  4.44570E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.71679E+02  7.33928E+02 ];
CPU_USAGE                 (idx, 1)        = 9.33248 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98993E+00 1.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.27454E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22664.88;
MEMSIZE                   (idx, 1)        = 22480.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 179 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.73265E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.69032E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.21895E+06 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.02780E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.40572E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.67128E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.51623E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.77414E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.23810E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.33662E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.92057E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.23919E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.34278E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.32963E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.31039E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30141E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.50073E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.56467E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.27984E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.18039E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.68459E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.90948E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.99472E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.89917E+15 0.00033  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32019E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.20958E-02 -2.60271E+27  2.17777E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02917E+00 0.00079 ];
TH232_FISS                (idx, [1:   4]) = [  1.65465E+18 0.00387  1.76533E-02 0.00382 ];
U233_FISS                 (idx, [1:   4]) = [  8.77906E+19 0.00053  9.36644E-01 0.00013 ];
U235_FISS                 (idx, [1:   4]) = [  2.48167E+18 0.00323  2.64777E-02 0.00322 ];
PU239_FISS                (idx, [1:   4]) = [  9.48840E+14 0.17108  1.01384E-05 0.17107 ];
TH232_CAPT                (idx, [1:   4]) = [  1.01723E+20 0.00057  7.06831E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.28434E+19 0.00140  8.92455E-02 0.00138 ];
U235_CAPT                 (idx, [1:   4]) = [  8.82656E+17 0.00582  6.13350E-03 0.00582 ];
U238_CAPT                 (idx, [1:   4]) = [  2.66562E+13 1.00000  1.86407E-07 1.00000 ];
PU239_CAPT                (idx, [1:   4]) = [  4.20248E+14 0.24367  2.92657E-06 0.24367 ];
PU240_CAPT                (idx, [1:   4]) = [  2.63762E+13 1.00000  1.83442E-07 1.00000 ];
SM149_CAPT                (idx, [1:   4]) = [  1.85261E+17 0.01172  1.28739E-03 0.01172 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001143 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14821E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001143 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5459884 5.46568E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3556106 3.55973E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 985153 9.86077E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001143 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.58325E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36265E+20 3.8E-06  2.36265E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37386E+19 1.1E-07  9.37386E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43866E+20 0.00030  1.27649E+20 0.00021  1.62165E+19 0.00179 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37605E+20 0.00018  2.21388E+20 0.00012  1.62165E+19 0.00179 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.63306E+20 0.00033  2.63306E+20 0.00033  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.67287E+22 0.00024  3.47758E+22 0.00022  1.95287E+21 0.00164 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.59646E+19 0.00135 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.63569E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19866E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.25179E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.25179E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.25402E-01 0.04533 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.38706E-01 0.01199 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.98688E-04 0.02364 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.13907E+03 0.02939 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01394E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 5.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.69750E-01 0.03131 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.93864E-01 0.03131 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52046E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99753E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.97306E-01 0.00047  9.93565E-02 0.00047  3.32395E-04 0.00936 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.97442E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.97331E-01 0.00033 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.97442E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.95633E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.00656E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.00445E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.35924E-02 0.00134 ];
IMP_EALF                  (idx, [1:   2]) = [  1.36192E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.71577E-01 0.00150 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.72085E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.77159E-03 0.00536  2.91361E-04 0.01885  9.04288E-04 0.01116  6.93676E-04 0.01247  1.49228E-03 0.00861  3.25873E-04 0.01846  6.41075E-05 0.04065 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.64500E-01 0.01519  1.24758E-02 2.9E-05  3.22943E-02 0.00011  1.06330E-01 0.00055  2.99067E-01 0.00028  1.24658E+00 0.00054  5.55092E+00 0.03371 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.35596E-03 0.00826  2.55193E-04 0.02751  7.99426E-04 0.01629  6.24582E-04 0.01952  1.32785E-03 0.01286  2.90164E-04 0.02847  5.87460E-05 0.06171 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.69746E-01 0.02287  1.24760E-02 3.4E-05  3.22935E-02 0.00015  1.06280E-01 0.00075  2.99078E-01 0.00041  1.24632E+00 0.00073  6.41631E+00 0.02701 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.73915E-07 0.00132  9.73536E-07 0.00132  1.08243E-06 0.01755 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.73840E-07 0.00123  8.73500E-07 0.00123  9.71137E-07 0.01752 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.33283E-03 0.00942  2.55650E-04 0.03095  8.13320E-04 0.01835  6.25529E-04 0.02310  1.28914E-03 0.01557  2.92220E-04 0.03209  5.69622E-05 0.06988 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.65692E-01 0.02442  1.24765E-02 4.7E-05  3.22892E-02 0.00019  1.06364E-01 0.00102  2.99351E-01 0.00053  1.24675E+00 0.00105  6.62439E+00 0.03769 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.61245E-07 0.01966  8.61155E-07 0.01966  8.97542E-07 0.05012 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.73002E-07 0.01965  7.72921E-07 0.01966  8.05409E-07 0.05012 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.97130E-03 0.04671  2.55582E-04 0.14640  6.93142E-04 0.07643  5.49620E-04 0.09426  1.11294E-03 0.06941  3.09626E-04 0.12671  5.03946E-05 0.31493 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.99873E-01 0.12662  1.24770E-02 0.00012  3.23008E-02 0.00060  1.06994E-01 0.00373  2.98577E-01 0.00189  1.24517E+00 0.00250  5.80447E+00 0.14536 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.98225E-03 0.04600  2.55139E-04 0.14205  6.95519E-04 0.07645  5.59887E-04 0.09263  1.10940E-03 0.06783  3.10633E-04 0.12457  5.16711E-05 0.32299 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.03745E-01 0.12586  1.24771E-02 0.00012  3.22998E-02 0.00060  1.06989E-01 0.00373  2.98584E-01 0.00188  1.24486E+00 0.00248  5.80447E+00 0.14536 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.45958E+03 0.04268 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.67944E-07 0.00080 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.68474E-07 0.00058 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.30647E-03 0.00633 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.41674E+03 0.00641 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.93208E-09 0.00078 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.68526E-05 0.00817  6.68909E-05 0.00816  1.45233E-06 0.38274 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.52969E-05 0.02471  5.51435E-05 0.02469  1.73824E-06 0.57667 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.88500E-04 0.02321  1.88514E-04 0.02330  1.94864E-04 0.37653 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71008E+01 0.01187 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.54741E+01 0.00025  4.06433E+01 0.00036 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 23:42:11 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 27 06:51:12 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651027331123 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00018E+00  1.00249E+00  1.00768E+00  1.00430E+00  9.96470E-01  1.00249E+00  1.00818E+00  9.94882E-01  9.99267E-01  9.94132E-01  9.98969E-01  1.00005E+00  1.00031E+00  1.00096E+00  9.93941E-01  1.00205E+00  9.96817E-01  9.96837E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  3.43263E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.65674E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30114E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44436E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.72041E+00 0.00028  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.54001E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.53022E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.62239E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.85644E+00 0.00046  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334282 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33397E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33397E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.99882E+03 ;
RUNNING_TIME              (idx, 1)        =  4.29020E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03138E+00  1.03138E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.27590E+00  1.49550E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.26437E+02  5.27428E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.50333E-01  1.89833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.91333E-02  1.43167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.01384E+01  4.31833E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.24744E+02  7.39438E+02 ];
CPU_USAGE                 (idx, 1)        = 9.32082 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98986E+00 1.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.26150E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22664.88;
MEMSIZE                   (idx, 1)        = 22480.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 180 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.89787E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73520E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.81465E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.03495E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.40507E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.82851E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56041E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.22065E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.78643E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.13896E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.28823E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.04671E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.85071E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.38265E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38276E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35091E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57175E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.85444E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.30656E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.28029E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.94246E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.49198E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.14104E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.93931E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64364E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.28321E-02 -4.91288E+27  2.20087E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03994E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.61786E+18 0.00422  1.72791E-02 0.00413 ];
U233_FISS                 (idx, [1:   4]) = [  8.43064E+19 0.00051  9.00467E-01 0.00017 ];
U235_FISS                 (idx, [1:   4]) = [  4.85915E+18 0.00231  5.19004E-02 0.00227 ];
PU239_FISS                (idx, [1:   4]) = [  2.65866E+16 0.03078  2.84033E-04 0.03076 ];
PU240_FISS                (idx, [1:   4]) = [  8.19716E+14 0.17037  8.74262E-06 0.17037 ];
PU241_FISS                (idx, [1:   4]) = [  7.92961E+14 0.18604  8.49296E-06 0.18594 ];
TH232_CAPT                (idx, [1:   4]) = [  9.95957E+19 0.00055  6.85145E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23048E+19 0.00155  8.46498E-02 0.00154 ];
U235_CAPT                 (idx, [1:   4]) = [  1.73775E+18 0.00405  1.19545E-02 0.00402 ];
U238_CAPT                 (idx, [1:   4]) = [  2.64160E+14 0.31146  1.82222E-06 0.31147 ];
PU239_CAPT                (idx, [1:   4]) = [  1.28018E+16 0.04492  8.80331E-05 0.04488 ];
PU240_CAPT                (idx, [1:   4]) = [  2.03568E+15 0.10991  1.39900E-05 0.10994 ];
PU241_CAPT                (idx, [1:   4]) = [  2.64228E+14 0.31144  1.81361E-06 0.31145 ];
SM149_CAPT                (idx, [1:   4]) = [  1.86660E+17 0.01156  1.28424E-03 0.01158 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001915 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.11211E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001915 1.00111E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5487583 5.49284E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3534734 3.53781E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 979598 9.80468E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001915 1.00111E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.88127E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36142E+20 4.1E-06  2.36142E+20 4.1E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36992E+19 1.1E-07  9.36992E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45335E+20 0.00027  1.29105E+20 0.00021  1.62301E+19 0.00172 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39034E+20 0.00017  2.22804E+20 0.00012  1.62301E+19 0.00172 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.64644E+20 0.00034  2.64644E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.68360E+22 0.00024  3.48828E+22 0.00023  1.95313E+21 0.00173 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.59484E+19 0.00149 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.64983E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20017E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.18564E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.18564E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.54794E-01 0.05450 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.29414E-01 0.01200 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.90213E-04 0.02500 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.07419E+03 0.03062 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01956E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.2E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.68798E-01 0.03998 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.03150E-01 0.03998 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52021E+00 4.0E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99837E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.91567E-01 0.00041  9.87271E-02 0.00041  3.39785E-04 0.00914 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.92171E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  8.92332E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.92171E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.89138E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.94652E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.94537E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.44343E-02 0.00146 ];
IMP_EALF                  (idx, [1:   2]) = [  1.44482E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.85158E-01 0.00157 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.85784E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.87671E-03 0.00544  2.86086E-04 0.01941  9.24542E-04 0.01086  6.95096E-04 0.01295  1.55023E-03 0.00867  3.44488E-04 0.01716  7.62750E-05 0.03897 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.87455E-01 0.01376  1.24759E-02 4.0E-05  3.22671E-02 0.00013  1.06569E-01 0.00059  3.00043E-01 0.00032  1.25349E+00 0.00070  6.02539E+00 0.02801 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.45321E-03 0.00798  2.51127E-04 0.02904  8.25576E-04 0.01650  6.11191E-04 0.01914  1.38554E-03 0.01290  3.11308E-04 0.02557  6.84713E-05 0.05342 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.90076E-01 0.02026  1.24770E-02 9.7E-05  3.22622E-02 0.00019  1.06698E-01 0.00084  3.00099E-01 0.00044  1.25360E+00 0.00092  6.57302E+00 0.02383 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.52291E-07 0.00128  9.51756E-07 0.00129  1.10207E-06 0.01857 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.48989E-07 0.00121  8.48511E-07 0.00122  9.82728E-07 0.01863 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.42134E-03 0.00932  2.48168E-04 0.03364  8.17866E-04 0.01821  5.99934E-04 0.02224  1.36341E-03 0.01468  3.21741E-04 0.03133  7.02229E-05 0.06183 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.98024E-01 0.02245  1.24754E-02 6.5E-05  3.22599E-02 0.00020  1.06590E-01 0.00102  2.99993E-01 0.00054  1.25389E+00 0.00162  6.69524E+00 0.03195 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.40046E-07 0.01969  8.39763E-07 0.01969  8.94376E-07 0.05233 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.48898E-07 0.01967  7.48644E-07 0.01967  7.97168E-07 0.05231 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.98736E-03 0.04355  2.57610E-04 0.15504  6.52923E-04 0.08380  5.33868E-04 0.09312  1.21743E-03 0.06469  2.75627E-04 0.12116  4.98969E-05 0.28914 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.90002E-01 0.09634  1.24770E-02 0.00014  3.22564E-02 0.00076  1.07045E-01 0.00377  3.00028E-01 0.00192  1.24805E+00 0.00472  7.12157E+00 0.10520 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.96455E-03 0.04303  2.55277E-04 0.15073  6.63104E-04 0.08347  5.27770E-04 0.09132  1.20163E-03 0.06445  2.70212E-04 0.11846  4.65552E-05 0.28137 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.87758E-01 0.09291  1.24770E-02 0.00013  3.22577E-02 0.00076  1.07037E-01 0.00374  3.00045E-01 0.00190  1.24777E+00 0.00471  7.12157E+00 0.10520 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.57588E+03 0.03902 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.43797E-07 0.00069 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.41420E-07 0.00058 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.37682E-03 0.00724 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.57816E+03 0.00723 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.90672E-09 0.00079 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.76006E-05 0.00894  6.75635E-05 0.00890  2.15280E-06 0.35573 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.84284E-05 0.02796  5.83202E-05 0.02805  1.97418E-06 0.37060 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.80170E-04 0.02383  1.79964E-04 0.02400  2.35975E-04 0.32968 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.64451E+01 0.01101 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.53022E+01 0.00026  4.00884E+01 0.00038 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 23:42:11 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 27 07:44:35 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651027331123 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00824E+00  9.97355E-01  9.96772E-01  9.97085E-01  1.00173E+00  1.00206E+00  9.94728E-01  9.98261E-01  9.91891E-01  9.95086E-01  9.96781E-01  1.00471E+00  1.00239E+00  1.00576E+00  1.00186E+00  1.00119E+00  1.00020E+00  1.00389E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  3.43840E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.65616E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30264E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44597E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.71908E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52976E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51996E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.59676E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.85672E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334028 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33373E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33373E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.49435E+03 ;
RUNNING_TIME              (idx, 1)        =  4.82404E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03138E+00  1.03138E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.42472E+00  1.48817E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.79626E+02  5.31887E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.68383E-01  1.80500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.91333E-02  1.43167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.42301E+01  4.09165E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.78354E+02  7.42902E+02 ];
CPU_USAGE                 (idx, 1)        = 9.31657 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98990E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.25667E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22664.88;
MEMSIZE                   (idx, 1)        = 22480.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 183 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.92579E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74226E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.24730E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.03002E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.40124E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.86118E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56770E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.21363E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.00083E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.03846E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.48593E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.13554E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.04442E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.81892E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38677E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35261E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57479E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.90108E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.73661E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.29296E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.99469E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.93588E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.15882E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.94320E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.30374E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.80417E-02 -6.03386E+27  2.21208E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03649E+00 0.00069 ];
TH232_FISS                (idx, [1:   4]) = [  1.61672E+18 0.00411  1.72566E-02 0.00401 ];
U233_FISS                 (idx, [1:   4]) = [  8.32487E+19 0.00054  8.88649E-01 0.00019 ];
U235_FISS                 (idx, [1:   4]) = [  5.56201E+18 0.00228  5.93722E-02 0.00222 ];
U238_FISS                 (idx, [1:   4]) = [  2.67513E+13 1.00000  2.85413E-07 1.00000 ];
PU239_FISS                (idx, [1:   4]) = [  5.64039E+16 0.02358  6.01943E-04 0.02353 ];
PU240_FISS                (idx, [1:   4]) = [  2.17372E+15 0.11180  2.31838E-05 0.11164 ];
PU241_FISS                (idx, [1:   4]) = [  2.75020E+15 0.09822  2.93682E-05 0.09822 ];
TH232_CAPT                (idx, [1:   4]) = [  9.83199E+19 0.00054  6.76172E-01 0.00034 ];
U233_CAPT                 (idx, [1:   4]) = [  1.21833E+19 0.00136  8.37893E-02 0.00135 ];
U235_CAPT                 (idx, [1:   4]) = [  1.97587E+18 0.00367  1.35884E-02 0.00363 ];
U238_CAPT                 (idx, [1:   4]) = [  2.37427E+14 0.36476  1.63281E-06 0.36564 ];
PU239_CAPT                (idx, [1:   4]) = [  2.55994E+16 0.03214  1.76026E-04 0.03213 ];
PU240_CAPT                (idx, [1:   4]) = [  6.56305E+15 0.06448  4.51550E-05 0.06458 ];
PU241_CAPT                (idx, [1:   4]) = [  4.50584E+14 0.23598  3.09892E-06 0.23598 ];
SM149_CAPT                (idx, [1:   4]) = [  1.84947E+17 0.01225  1.27200E-03 0.01225 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001185 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10020E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001185 1.00110E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5486265 5.49177E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3534743 3.53816E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 980177 9.81073E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001185 1.00110E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.43424E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36138E+20 4.3E-06  2.36138E+20 4.3E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36848E+19 1.0E-07  9.36848E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45372E+20 0.00028  1.29055E+20 0.00020  1.63170E+19 0.00176 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39057E+20 0.00017  2.22740E+20 0.00012  1.63170E+19 0.00176 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.64773E+20 0.00036  2.64773E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.67859E+22 0.00025  3.48260E+22 0.00022  1.95991E+21 0.00173 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.59772E+19 0.00140 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.65034E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19801E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.15124E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.15124E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.49656E-01 0.04779 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.35197E-01 0.01161 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.01151E-04 0.02567 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.37293E+03 0.02874 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01896E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 7.2E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.28315E-01 0.03458 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.56909E-01 0.03458 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52056E+00 4.2E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99867E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.91707E-01 0.00043  9.87563E-02 0.00044  3.34068E-04 0.00957 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.91970E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  8.91885E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.91970E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.89012E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.91881E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.92226E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.48394E-02 0.00141 ];
IMP_EALF                  (idx, [1:   2]) = [  1.47860E-02 0.00096 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.91248E-01 0.00155 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.90621E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.93274E-03 0.00542  2.95972E-04 0.02114  9.02922E-04 0.01099  7.20368E-04 0.01211  1.58166E-03 0.00848  3.54799E-04 0.01797  7.70165E-05 0.03754 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.00248E-01 0.01435  1.24769E-02 4.9E-05  3.22494E-02 0.00013  1.06687E-01 0.00061  3.00583E-01 0.00028  1.25796E+00 0.00072  6.46200E+00 0.02686 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.40444E-03 0.00788  2.47235E-04 0.03093  7.88155E-04 0.01586  6.30614E-04 0.01870  1.37490E-03 0.01254  2.96867E-04 0.02525  6.66643E-05 0.05746 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.97124E-01 0.02242  1.24761E-02 4.8E-05  3.22453E-02 0.00017  1.06782E-01 0.00086  3.00583E-01 0.00042  1.25876E+00 0.00103  7.06761E+00 0.02208 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.40309E-07 0.00133  9.39911E-07 0.00133  1.05015E-06 0.02964 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.38420E-07 0.00121  8.38065E-07 0.00122  9.36236E-07 0.02951 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.37839E-03 0.00943  2.49119E-04 0.03756  7.69520E-04 0.02048  6.32889E-04 0.02049  1.34978E-03 0.01541  3.10867E-04 0.02976  6.62230E-05 0.06885 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.05018E-01 0.02756  1.24759E-02 7.1E-05  3.22505E-02 0.00025  1.06578E-01 0.00100  3.00636E-01 0.00054  1.25858E+00 0.00139  7.04087E+00 0.03150 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.26971E-07 0.01967  8.26735E-07 0.01967  8.89922E-07 0.06090 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.37545E-07 0.01967  7.37335E-07 0.01967  7.93677E-07 0.06090 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.11825E-03 0.04210  2.60048E-04 0.14684  6.68625E-04 0.08359  5.19259E-04 0.09129  1.23343E-03 0.06146  3.51057E-04 0.11980  8.58325E-05 0.23803 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.53961E-01 0.10118  1.24789E-02 9.3E-05  3.22141E-02 0.00077  1.06553E-01 0.00324  3.01664E-01 0.00208  1.25793E+00 0.00370  7.35818E+00 0.08100 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.14994E-03 0.04158  2.56764E-04 0.14384  6.78349E-04 0.08189  5.16572E-04 0.08828  1.25748E-03 0.06153  3.50001E-04 0.11612  9.07800E-05 0.23400 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.63683E-01 0.09980  1.24788E-02 9.5E-05  3.22153E-02 0.00076  1.06570E-01 0.00325  3.01627E-01 0.00207  1.25801E+00 0.00369  7.35818E+00 0.08100 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.79577E+03 0.03808 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.32891E-07 0.00083 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.31826E-07 0.00074 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.32116E-03 0.00657 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.56105E+03 0.00666 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.89947E-09 0.00086 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.70067E-05 0.00907  6.70074E-05 0.00906  1.06488E-06 0.48652 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.43746E-05 0.02546  5.43520E-05 0.02554  1.05784E-06 0.47195 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.93543E-04 0.02471  1.93806E-04 0.02465  1.19696E-04 0.44607 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.61058E+01 0.01122 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51996E+01 0.00027  3.98393E+01 0.00038 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 23:42:11 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 27 08:38:41 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651027331123 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.00002E+00  9.96944E-01  1.00110E+00  1.00306E+00  9.96660E-01  9.96940E-01  9.96313E-01  1.00067E+00  1.00225E+00  1.00349E+00  9.98125E-01  1.00275E+00  9.98845E-01  1.00513E+00  9.99212E-01  9.98287E-01  1.00181E+00  9.98375E-01  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 0.0E+00  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  3.44562E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.65544E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.30304E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44660E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.72216E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52471E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51491E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.58472E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.85943E+00 0.00047  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334600 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33436E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33436E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.99664E+03 ;
RUNNING_TIME              (idx, 1)        =  5.36503E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03138E+00  1.03138E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.57258E+00  1.47867E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.33528E+02  5.39020E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.86783E-01  1.84000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.91333E-02  1.43167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.83619E+01  4.13177E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.32410E+02  7.45830E+02 ];
CPU_USAGE                 (idx, 1)        = 9.31334 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98998E+00 1.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.25300E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22664.88;
MEMSIZE                   (idx, 1)        = 22480.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 184 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.92332E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74220E+08 ;
TOT_SF_RATE               (idx, 1)        =  8.84531E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01461E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39062E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.87414E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56873E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.32763E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.18122E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.07289E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.65986E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.21222E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.20684E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.20342E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37719E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34424E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56390E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.92488E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.11857E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.28360E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01463E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.28382E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.13578E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.94229E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.96384E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -3.33211E-02 -7.16985E+27  2.22344E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03471E+00 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.59128E+18 0.00458  1.69896E-02 0.00451 ];
U233_FISS                 (idx, [1:   4]) = [  8.24552E+19 0.00057  8.80394E-01 0.00019 ];
U235_FISS                 (idx, [1:   4]) = [  6.03250E+18 0.00207  6.44137E-02 0.00208 ];
U238_FISS                 (idx, [1:   4]) = [  5.29561E+13 0.70592  5.61671E-07 0.70596 ];
PU239_FISS                (idx, [1:   4]) = [  1.00278E+17 0.01659  1.07066E-03 0.01657 ];
PU240_FISS                (idx, [1:   4]) = [  4.92609E+15 0.07564  5.26162E-05 0.07559 ];
PU241_FISS                (idx, [1:   4]) = [  5.98686E+15 0.06045  6.39378E-05 0.06046 ];
TH232_CAPT                (idx, [1:   4]) = [  9.73654E+19 0.00054  6.69679E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.20431E+19 0.00152  8.28326E-02 0.00146 ];
U235_CAPT                 (idx, [1:   4]) = [  2.13236E+18 0.00385  1.46666E-02 0.00384 ];
U238_CAPT                 (idx, [1:   4]) = [  4.25083E+14 0.24368  2.92318E-06 0.24367 ];
PU239_CAPT                (idx, [1:   4]) = [  4.68384E+16 0.02232  3.22176E-04 0.02234 ];
PU240_CAPT                (idx, [1:   4]) = [  1.32502E+16 0.04052  9.11428E-05 0.04055 ];
PU241_CAPT                (idx, [1:   4]) = [  1.32431E+15 0.14110  9.10300E-06 0.14110 ];
SM149_CAPT                (idx, [1:   4]) = [  1.90730E+17 0.01186  1.31180E-03 0.01185 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10003087 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10073E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10003087 1.00110E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5487262 5.49181E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3534994 3.53767E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 980831 9.81527E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10003087 1.00110E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 8.38190E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36157E+20 4.2E-06  2.36157E+20 4.2E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36720E+19 1.0E-07  9.36720E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45335E+20 0.00027  1.29021E+20 0.00021  1.63144E+19 0.00167 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39007E+20 0.00017  2.22693E+20 0.00012  1.63144E+19 0.00167 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.64743E+20 0.00034  2.64743E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.67655E+22 0.00023  3.47985E+22 0.00021  1.96702E+21 0.00170 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.59860E+19 0.00130 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.64993E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19654E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  8.11758E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.11758E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.56880E-01 0.04950 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.33643E-01 0.01202 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.97584E-04 0.02458 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.43616E+03 0.02850 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01849E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 4.8E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.25117E-01 0.03488 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.53942E-01 0.03488 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52111E+00 4.1E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99895E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.91945E-01 0.00045  9.87614E-02 0.00043  3.38294E-04 0.00950 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.92177E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  8.92056E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.92177E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.89295E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.90440E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.90239E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.50549E-02 0.00142 ];
IMP_EALF                  (idx, [1:   2]) = [  1.50825E-02 0.00091 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.94744E-01 0.00152 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.94518E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.97220E-03 0.00539  2.90334E-04 0.01961  9.34723E-04 0.01167  7.11730E-04 0.01270  1.59906E-03 0.00838  3.56247E-04 0.01722  8.01096E-05 0.03957 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.00458E-01 0.01524  1.24760E-02 3.8E-05  3.22446E-02 0.00014  1.06635E-01 0.00055  3.00624E-01 0.00030  1.25744E+00 0.00095  6.21444E+00 0.02674 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.46722E-03 0.00755  2.64859E-04 0.03119  8.17515E-04 0.01627  6.22003E-04 0.01843  1.38070E-03 0.01222  3.13316E-04 0.02612  6.88262E-05 0.06021 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.00542E-01 0.02400  1.24757E-02 4.3E-05  3.22421E-02 0.00017  1.06573E-01 0.00073  3.00575E-01 0.00043  1.25677E+00 0.00143  6.82722E+00 0.02196 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.33119E-07 0.00140  9.32713E-07 0.00140  1.05511E-06 0.01676 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.32239E-07 0.00131  8.31876E-07 0.00132  9.41242E-07 0.01680 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.41496E-03 0.00963  2.44020E-04 0.03392  8.17869E-04 0.02017  6.17341E-04 0.02282  1.36809E-03 0.01424  3.02563E-04 0.02864  6.50739E-05 0.06704 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.86562E-01 0.02493  1.24770E-02 0.00012  3.22448E-02 0.00021  1.06591E-01 0.00100  3.00719E-01 0.00051  1.26001E+00 0.00136  6.77683E+00 0.03355 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.14936E-07 0.01958  8.14492E-07 0.01959  9.70352E-07 0.06010 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.27020E-07 0.01958  7.26622E-07 0.01958  8.65978E-07 0.06009 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.08727E-03 0.04361  2.14445E-04 0.13589  6.78338E-04 0.08673  5.49121E-04 0.09797  1.27172E-03 0.06267  2.89185E-04 0.12213  8.44639E-05 0.23522 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.76003E-01 0.10131  1.24770E-02 0.00014  3.22421E-02 0.00058  1.07255E-01 0.00367  3.01472E-01 0.00209  1.26044E+00 0.00389  7.44340E+00 0.07506 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.07964E-03 0.04373  2.15006E-04 0.13507  6.75699E-04 0.08583  5.52641E-04 0.09524  1.26191E-03 0.06291  2.93140E-04 0.12017  8.12513E-05 0.22747 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.73096E-01 0.10238  1.24770E-02 0.00014  3.22398E-02 0.00058  1.07265E-01 0.00368  3.01471E-01 0.00208  1.26032E+00 0.00390  7.44340E+00 0.07506 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.79830E+03 0.03906 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.24359E-07 0.00084 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.24420E-07 0.00066 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.47214E-03 0.00636 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.75773E+03 0.00651 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.89164E-09 0.00079 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.76000E-05 0.00736  6.75726E-05 0.00739  1.39314E-06 0.50714 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.53245E-05 0.02594  5.52579E-05 0.02589  1.76334E-06 0.62985 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.90572E-04 0.02295  1.90830E-04 0.02294  1.34784E-04 0.44765 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.65755E+01 0.01150 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51491E+01 0.00027  3.97064E+01 0.00039 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 23:42:11 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 27 09:33:17 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651027331123 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  9.98223E-01  1.00117E+00  1.00117E+00  9.96106E-01  9.94131E-01  1.00181E+00  9.99536E-01  1.00385E+00  9.97179E-01  1.00581E+00  1.00030E+00  9.99027E-01  9.99904E-01  1.00198E+00  9.97268E-01  9.95866E-01  1.00412E+00  1.00254E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 1.6E-09  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  3.47001E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.65300E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.29823E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.44267E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.72577E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50928E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49946E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.57025E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.87326E+00 0.00045  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333567 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33395E+04 0.00065 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33395E+04 0.00065 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.50378E+03 ;
RUNNING_TIME              (idx, 1)        =  5.91098E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03138E+00  1.03138E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.72605E+00  1.53467E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.87918E+02  5.43897E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.07300E-01  2.05167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.91333E-02  1.43167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.24987E+01  4.13663E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.87002E+02  7.49230E+02 ];
CPU_USAGE                 (idx, 1)        = 9.31111 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99008E+00 1.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.25054E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22664.88;
MEMSIZE                   (idx, 1)        = 22480.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 188 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.94054E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74759E+08 ;
TOT_SF_RATE               (idx, 1)        =  8.11269E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.97645E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.36782E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.92953E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57642E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.81208E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.75833E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  8.32169E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.04526E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44254E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.70363E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34146E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37206E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33772E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55647E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.96806E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.27159E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27670E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.06485E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.42212E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.10506E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.96502E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.60423E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -5.43301E-02 -1.16905E+28  2.26865E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.03104E+00 0.00076 ];
TH232_FISS                (idx, [1:   4]) = [  1.57013E+18 0.00416  1.67545E-02 0.00415 ];
U233_FISS                 (idx, [1:   4]) = [  8.05585E+19 0.00055  8.59603E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  6.92413E+18 0.00203  7.38833E-02 0.00195 ];
U238_FISS                 (idx, [1:   4]) = [  1.05571E+14 0.49750  1.12041E-06 0.49749 ];
PU239_FISS                (idx, [1:   4]) = [  3.34246E+17 0.00883  3.56610E-03 0.00877 ];
PU240_FISS                (idx, [1:   4]) = [  2.63917E+16 0.03062  2.81636E-04 0.03063 ];
PU241_FISS                (idx, [1:   4]) = [  3.83177E+16 0.02646  4.08755E-04 0.02641 ];
TH232_CAPT                (idx, [1:   4]) = [  9.52098E+19 0.00058  6.52264E-01 0.00035 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17222E+19 0.00148  8.03099E-02 0.00150 ];
U235_CAPT                 (idx, [1:   4]) = [  2.44694E+18 0.00309  1.67646E-02 0.00312 ];
U238_CAPT                 (idx, [1:   4]) = [  1.16668E+15 0.14312  7.99503E-06 0.14313 ];
PU239_CAPT                (idx, [1:   4]) = [  1.58381E+17 0.01247  1.08505E-03 0.01247 ];
PU240_CAPT                (idx, [1:   4]) = [  6.63470E+16 0.02006  4.54501E-04 0.02003 ];
PU241_CAPT                (idx, [1:   4]) = [  9.00642E+15 0.05317  6.17221E-05 0.05319 ];
SM149_CAPT                (idx, [1:   4]) = [  1.84822E+17 0.01159  1.26617E-03 0.01159 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001853 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08893E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001853 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5492674 5.49784E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3526751 3.52981E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 982428 9.83235E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001853 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.58908E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36329E+20 4.4E-06  2.36329E+20 4.4E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36324E+19 1.0E-07  9.36324E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.45974E+20 0.00027  1.29477E+20 0.00020  1.64960E+19 0.00176 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39606E+20 0.00017  2.23110E+20 0.00011  1.64960E+19 0.00176 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.65501E+20 0.00034  2.65501E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.68625E+22 0.00024  3.48861E+22 0.00022  1.97638E+21 0.00165 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.61057E+19 0.00147 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.65712E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19585E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  7.99190E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.99190E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.79007E-01 0.04746 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.37229E-01 0.01183 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.23735E-04 0.02497 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.17793E+03 0.02916 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.01680E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.8E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.57248E-01 0.03191 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.82859E-01 0.03191 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52401E+00 4.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99979E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.90925E-01 0.00046  9.86505E-02 0.00045  3.44883E-04 0.00875 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.90407E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  8.90159E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.90407E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  9.87533E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.85753E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.86065E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.57770E-02 0.00135 ];
IMP_EALF                  (idx, [1:   2]) = [  1.57255E-02 0.00093 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.04866E-01 0.00151 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.04126E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.03319E-03 0.00515  2.92927E-04 0.01966  9.28783E-04 0.01076  7.26011E-04 0.01208  1.63281E-03 0.00806  3.72774E-04 0.01694  7.98822E-05 0.03940 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.05318E-01 0.01406  1.24769E-02 4.1E-05  3.22254E-02 0.00014  1.06731E-01 0.00052  3.01181E-01 0.00032  1.25958E+00 0.00086  6.32076E+00 0.02758 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.51072E-03 0.00795  2.55158E-04 0.02986  7.95636E-04 0.01669  6.34933E-04 0.01828  1.42886E-03 0.01211  3.20748E-04 0.02568  7.53824E-05 0.05750 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.14599E-01 0.02235  1.24766E-02 4.8E-05  3.22214E-02 0.00019  1.06721E-01 0.00075  3.01102E-01 0.00046  1.26050E+00 0.00115  7.13530E+00 0.02017 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.15998E-07 0.00123  9.15556E-07 0.00123  1.04127E-06 0.02042 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.16030E-07 0.00113  8.15636E-07 0.00113  9.27849E-07 0.02048 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.49063E-03 0.00891  2.56983E-04 0.03302  7.86683E-04 0.01915  6.31650E-04 0.02248  1.41560E-03 0.01443  3.32199E-04 0.02824  6.75188E-05 0.06316 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.02316E-01 0.02223  1.24753E-02 6.2E-05  3.22166E-02 0.00029  1.06770E-01 0.00106  3.00965E-01 0.00054  1.25652E+00 0.00155  7.16974E+00 0.02943 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.09874E-07 0.01976  8.09640E-07 0.01977  8.46831E-07 0.05528 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.21625E-07 0.01974  7.21416E-07 0.01974  7.54539E-07 0.05538 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.17235E-03 0.04410  2.20210E-04 0.15591  7.15789E-04 0.08211  5.84522E-04 0.09046  1.27667E-03 0.06254  3.21156E-04 0.11146  5.40095E-05 0.29102 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.87652E-01 0.08982  1.24730E-02 0.00022  3.22000E-02 0.00089  1.06471E-01 0.00287  3.00557E-01 0.00186  1.24648E+00 0.00424  7.60143E+00 0.08759 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.16624E-03 0.04399  2.29214E-04 0.15397  7.18090E-04 0.08050  5.78698E-04 0.09124  1.27086E-03 0.06190  3.14803E-04 0.10857  5.45788E-05 0.29844 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.88304E-01 0.08950  1.24733E-02 0.00022  3.22004E-02 0.00088  1.06471E-01 0.00283  3.00579E-01 0.00186  1.24642E+00 0.00424  7.57809E+00 0.08765 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.96563E+03 0.04039 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.09090E-07 0.00077 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.09880E-07 0.00062 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.56791E-03 0.00518 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.92481E+03 0.00514 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.93053E-09 0.00088 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.73316E-05 0.00835  6.72902E-05 0.00837  1.93519E-06 0.35640 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.96387E-05 0.02692  5.94576E-05 0.02707  2.32498E-06 0.40351 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.15237E-04 0.02449  2.15305E-04 0.02463  2.00198E-04 0.35142 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.62560E+01 0.01118 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49946E+01 0.00026  3.92607E+01 0.00037 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 23:42:11 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 27 10:28:10 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651027331123 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  9.98928E-01  1.00705E+00  1.00526E+00  1.00024E+00  9.89346E-01  9.98458E-01  1.00290E+00  1.00426E+00  1.00045E+00  9.97082E-01  1.00007E+00  9.99164E-01  9.98948E-01  9.99418E-01  9.93363E-01  9.99776E-01  1.00037E+00  1.00493E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 1.3E-09  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  3.53570E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.64643E-01 1.2E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.28209E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.42898E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.73964E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50501E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49506E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.61406E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.92794E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333816 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33399E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33399E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.01365E+03 ;
RUNNING_TIME              (idx, 1)        =  6.45992E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03138E+00  1.03138E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.88567E+00  1.59617E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.42597E+02  5.46793E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.26067E-01  1.87667E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.91333E-02  1.43167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.66559E+01  4.15722E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.41877E+02  7.51013E+02 ];
CPU_USAGE                 (idx, 1)        = 9.30917 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98994E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.24843E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22664.88;
MEMSIZE                   (idx, 1)        = 22480.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 190 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.92342E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74185E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.98935E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92993E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.34296E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95893E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57317E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.49275E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.38467E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.31806E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.13506E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69258E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.23891E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.57885E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34649E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31730E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52906E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.00318E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.59828E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.24086E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.04181E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.11011E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.03260E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.97956E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.32052E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.09047E-01 -2.34641E+28  2.38639E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02528E+00 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.52611E+18 0.00413  1.63137E-02 0.00404 ];
U233_FISS                 (idx, [1:   4]) = [  7.92317E+19 0.00053  8.47028E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.12525E+18 0.00194  7.61721E-02 0.00186 ];
U238_FISS                 (idx, [1:   4]) = [  2.39332E+14 0.32886  2.55626E-06 0.32886 ];
PU239_FISS                (idx, [1:   4]) = [  6.24093E+17 0.00605  6.67214E-03 0.00605 ];
PU240_FISS                (idx, [1:   4]) = [  7.72968E+16 0.01789  8.26445E-04 0.01791 ];
PU241_FISS                (idx, [1:   4]) = [  1.33074E+17 0.01447  1.42266E-03 0.01449 ];
TH232_CAPT                (idx, [1:   4]) = [  9.32661E+19 0.00056  6.37684E-01 0.00033 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15581E+19 0.00151  7.90263E-02 0.00146 ];
U235_CAPT                 (idx, [1:   4]) = [  2.49834E+18 0.00317  1.70816E-02 0.00313 ];
U238_CAPT                 (idx, [1:   4]) = [  3.05418E+15 0.09799  2.09017E-05 0.09808 ];
PU239_CAPT                (idx, [1:   4]) = [  2.89161E+17 0.00907  1.97730E-03 0.00909 ];
PU240_CAPT                (idx, [1:   4]) = [  1.96686E+17 0.01227  1.34468E-03 0.01223 ];
PU241_CAPT                (idx, [1:   4]) = [  2.74270E+16 0.03089  1.87567E-04 0.03092 ];
SM149_CAPT                (idx, [1:   4]) = [  1.95278E+17 0.01207  1.33502E-03 0.01204 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001979 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08178E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001979 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5493665 5.49872E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3513837 3.51681E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 994477 9.95290E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001979 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 7.63685E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36625E+20 5.0E-06  2.36625E+20 5.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35945E+19 1.1E-07  9.35945E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46326E+20 0.00029  1.29434E+20 0.00019  1.68924E+19 0.00178 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.39921E+20 0.00018  2.23028E+20 0.00011  1.68924E+19 0.00178 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.65985E+20 0.00035  2.65985E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.71733E+22 0.00026  3.51782E+22 0.00023  1.99505E+21 0.00180 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.64741E+19 0.00146 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.66395E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.19685E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  7.74265E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.74265E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.89655E-01 0.04387 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.27715E-01 0.00964 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.85784E-04 0.02000 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.37985E+03 0.03178 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.00476E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99994E-01 7.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  7.86613E-01 0.02922 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.08368E-01 0.02922 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52819E+00 5.0E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00060E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.88979E-01 0.00043  9.84511E-02 0.00044  3.42261E-04 0.00888 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.89222E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.89650E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.89222E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.87458E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.83358E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.83489E+00 0.00015 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.61596E-02 0.00140 ];
IMP_EALF                  (idx, [1:   2]) = [  1.61362E-02 0.00100 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.10498E-01 0.00147 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.10619E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.02781E-03 0.00572  2.90933E-04 0.01861  9.48262E-04 0.01085  7.31906E-04 0.01370  1.60426E-03 0.00820  3.65892E-04 0.01819  8.65604E-05 0.03626 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.12937E-01 0.01468  1.24784E-02 7.7E-05  3.22041E-02 0.00014  1.06692E-01 0.00051  3.01427E-01 0.00028  1.25750E+00 0.00110  6.47593E+00 0.02445 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.47368E-03 0.00810  2.57340E-04 0.02678  8.10721E-04 0.01609  6.19929E-04 0.01837  1.39384E-03 0.01219  3.17647E-04 0.02477  7.42012E-05 0.05592 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.11932E-01 0.02149  1.24778E-02 4.9E-05  3.22050E-02 0.00022  1.06627E-01 0.00069  3.01607E-01 0.00043  1.26014E+00 0.00129  6.85884E+00 0.02125 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.14195E-07 0.00139  9.13781E-07 0.00140  1.02693E-06 0.01838 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.12646E-07 0.00130  8.12279E-07 0.00130  9.13031E-07 0.01836 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.46102E-03 0.00906  2.42329E-04 0.03362  8.05771E-04 0.01993  6.18808E-04 0.02218  1.39997E-03 0.01409  3.20276E-04 0.03005  7.38661E-05 0.06104 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.15792E-01 0.02323  1.24775E-02 8.2E-05  3.22054E-02 0.00028  1.06786E-01 0.00102  3.01485E-01 0.00052  1.25320E+00 0.00252  7.10954E+00 0.02875 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.01743E-07 0.01973  8.01366E-07 0.01973  9.48078E-07 0.05812 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.12852E-07 0.01972  7.12517E-07 0.01972  8.42969E-07 0.05808 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.03334E-03 0.04472  2.26142E-04 0.13016  6.71781E-04 0.08433  3.92959E-04 0.09965  1.37676E-03 0.06105  2.77349E-04 0.11640  8.83431E-05 0.21512 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  5.08541E-01 0.12028  1.24902E-02 0.00093  3.21724E-02 0.00090  1.07186E-01 0.00356  3.01338E-01 0.00204  1.25679E+00 0.00523  6.22473E+00 0.08596 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.02816E-03 0.04371  2.20074E-04 0.12741  6.66837E-04 0.08218  3.95722E-04 0.09870  1.36153E-03 0.05960  2.92368E-04 0.11652  9.16257E-05 0.21252 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  5.09457E-01 0.11659  1.24902E-02 0.00093  3.21723E-02 0.00089  1.07162E-01 0.00351  3.01379E-01 0.00204  1.25720E+00 0.00521  6.22474E+00 0.08596 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.83265E+03 0.04111 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.05934E-07 0.00078 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.05308E-07 0.00063 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.45404E-03 0.00531 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.81287E+03 0.00528 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.07805E-09 0.00091 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.84170E-05 0.00556  6.84267E-05 0.00555  1.23368E-06 0.41988 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.83617E-05 0.02036  5.83150E-05 0.02042  9.00660E-07 0.68938 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.74439E-04 0.01980  2.74943E-04 0.01977  1.51370E-04 0.40594 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63508E+01 0.01053 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49506E+01 0.00027  3.89890E+01 0.00036 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 23:42:11 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 27 11:23:21 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651027331123 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01145E+00  1.00510E+00  9.99507E-01  1.00383E+00  9.93273E-01  9.95292E-01  9.97738E-01  1.00168E+00  9.99634E-01  9.98571E-01  9.94342E-01  1.00168E+00  9.94498E-01  1.00260E+00  9.96777E-01  1.00028E+00  9.94788E-01  1.00895E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 1.3E-09  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  3.60713E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63929E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.26934E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.41883E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.75218E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51529E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50529E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.67264E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.98990E+00 0.00048  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333425 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33342E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33342E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.52549E+03 ;
RUNNING_TIME              (idx, 1)        =  7.01179E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03138E+00  1.03138E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.04913E+00  1.63467E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.97563E+02  5.49656E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.44633E-01  1.85833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.91333E-02  1.43167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.09145E+01  4.25852E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.96966E+02  7.51827E+02 ];
CPU_USAGE                 (idx, 1)        = 9.30645 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98986E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.24547E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22664.88;
MEMSIZE                   (idx, 1)        = 22480.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 190 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.91818E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74004E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.38854E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91587E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33506E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96764E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57204E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.57438E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.55472E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.39215E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14718E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76500E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.39625E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.93097E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33972E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31233E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52210E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.01961E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.00892E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22891E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.03157E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.27091E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.00865E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.99467E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.98062E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.64983E-01 -3.55002E+28  2.50675E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02143E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.53944E+18 0.00416  1.64618E-02 0.00409 ];
U233_FISS                 (idx, [1:   4]) = [  7.91119E+19 0.00053  8.46015E-01 0.00021 ];
U235_FISS                 (idx, [1:   4]) = [  7.08926E+18 0.00181  7.58106E-02 0.00171 ];
U238_FISS                 (idx, [1:   4]) = [  2.12600E+14 0.34940  2.27228E-06 0.34940 ];
PU239_FISS                (idx, [1:   4]) = [  6.60327E+17 0.00608  7.06163E-03 0.00608 ];
PU240_FISS                (idx, [1:   4]) = [  8.63124E+16 0.01808  9.23064E-04 0.01808 ];
PU241_FISS                (idx, [1:   4]) = [  1.57494E+17 0.01290  1.68414E-03 0.01287 ];
TH232_CAPT                (idx, [1:   4]) = [  9.26984E+19 0.00054  6.32445E-01 0.00035 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15209E+19 0.00158  7.86036E-02 0.00156 ];
U235_CAPT                 (idx, [1:   4]) = [  2.50028E+18 0.00331  1.70583E-02 0.00328 ];
U238_CAPT                 (idx, [1:   4]) = [  2.85271E+15 0.09219  1.94489E-05 0.09215 ];
PU239_CAPT                (idx, [1:   4]) = [  3.07141E+17 0.00880  2.09536E-03 0.00876 ];
PU240_CAPT                (idx, [1:   4]) = [  2.33247E+17 0.01117  1.59142E-03 0.01118 ];
PU241_CAPT                (idx, [1:   4]) = [  3.34753E+16 0.03003  2.28478E-04 0.03008 ];
SM149_CAPT                (idx, [1:   4]) = [  2.00532E+17 0.01144  1.36791E-03 0.01138 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000260 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08273E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000260 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5494161 5.50013E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3505417 3.50904E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1000682 1.00166E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000260 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.38419E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36679E+20 5.0E-06  2.36679E+20 5.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35894E+19 1.1E-07  9.35894E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46540E+20 0.00030  1.29026E+20 0.00019  1.75139E+19 0.00176 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.40129E+20 0.00018  2.22615E+20 0.00011  1.75139E+19 0.00176 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.66489E+20 0.00036  2.66489E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.75575E+22 0.00026  3.55195E+22 0.00022  2.03797E+21 0.00168 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.66941E+19 0.00144 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.66823E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20185E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  7.53670E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.53670E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.23263E-01 0.03762 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.07199E-01 0.00882 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.66691E-04 0.01711 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  6.50337E+03 0.03467 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.99840E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99993E-01 8.9E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.67933E-01 0.02136 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.81029E-01 0.02136 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52891E+00 4.9E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00071E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.87478E-01 0.00043  9.82582E-02 0.00042  3.41506E-04 0.00867 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.87995E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.88171E-01 0.00036 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.87995E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.86824E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.83510E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.83440E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.61353E-02 0.00143 ];
IMP_EALF                  (idx, [1:   2]) = [  1.61435E-02 0.00088 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.11382E-01 0.00146 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.11313E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.01544E-03 0.00518  2.78795E-04 0.02080  9.37891E-04 0.01168  7.07972E-04 0.01356  1.62680E-03 0.00827  3.81684E-04 0.01569  8.22984E-05 0.03798 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.08955E-01 0.01449  1.24775E-02 5.4E-05  3.21957E-02 0.00015  1.06707E-01 0.00048  3.01604E-01 0.00030  1.25901E+00 0.00096  6.04702E+00 0.02841 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.49501E-03 0.00775  2.39269E-04 0.03027  8.19698E-04 0.01574  6.09275E-04 0.01920  1.42957E-03 0.01118  3.29360E-04 0.02440  6.78369E-05 0.05054 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.04627E-01 0.01973  1.24772E-02 3.6E-05  3.21936E-02 0.00021  1.06687E-01 0.00067  3.01546E-01 0.00043  1.26065E+00 0.00115  6.81476E+00 0.02256 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.29522E-07 0.00144  9.29020E-07 0.00144  1.06962E-06 0.02394 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.24881E-07 0.00136  8.24435E-07 0.00136  9.49246E-07 0.02388 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.45935E-03 0.00889  2.30409E-04 0.03625  8.16791E-04 0.01832  6.15323E-04 0.02201  1.39853E-03 0.01437  3.35575E-04 0.02661  6.27214E-05 0.06681 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.93883E-01 0.02462  1.24760E-02 6.4E-05  3.21980E-02 0.00026  1.06737E-01 0.00112  3.01748E-01 0.00053  1.25896E+00 0.00173  6.62241E+00 0.03486 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.09574E-07 0.01977  8.09310E-07 0.01977  8.44632E-07 0.05151 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.18765E-07 0.01977  7.18530E-07 0.01978  7.49887E-07 0.05156 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.09293E-03 0.04043  2.12563E-04 0.15706  7.42590E-04 0.07710  4.90746E-04 0.09520  1.29995E-03 0.06550  3.11596E-04 0.13036  3.54823E-05 0.29274 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.44254E-01 0.07249  1.24791E-02 9.3E-05  3.21898E-02 0.00082  1.07002E-01 0.00358  3.01273E-01 0.00191  1.27061E+00 0.00444  5.29940E+00 0.12186 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.08534E-03 0.03992  2.15997E-04 0.15907  7.30805E-04 0.07691  4.95654E-04 0.09276  1.28915E-03 0.06327  3.17444E-04 0.12784  3.62973E-05 0.28306 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.43116E-01 0.06731  1.24791E-02 9.3E-05  3.21882E-02 0.00082  1.06976E-01 0.00353  3.01323E-01 0.00191  1.27037E+00 0.00444  5.29940E+00 0.12186 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.84107E+03 0.03574 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.19186E-07 0.00078 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.15711E-07 0.00064 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.47913E-03 0.00475 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.78587E+03 0.00484 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.27536E-09 0.00091 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.95311E-05 0.00489  6.95414E-05 0.00488  3.11624E-06 0.27160 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.87223E-05 0.01590  5.87237E-05 0.01595  2.57479E-06 0.32609 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.51163E-04 0.01602  3.51075E-04 0.01615  3.66936E-04 0.27210 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60567E+01 0.01112 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50529E+01 0.00028  3.90244E+01 0.00037 ];


% Increase counter:

if (exist('idx', 'var'));
  idx = idx + 1;
else;
  idx = 1;
end;

% Version, title and date:

VERSION                   (idx, [1: 14])  = 'Serpent 2.1.32' ;
COMPILE_DATE              (idx, [1: 20])  = 'Jun 15 2021 18:34:20' ;
DEBUG                     (idx, 1)        = 0 ;
TITLE                     (idx, [1:  8])  = 'Untitled' ;
CONFIDENTIAL_DATA         (idx, 1)        = 0 ;
INPUT_FILE_NAME           (idx, [1: 32])  = 'msfr_mix1_benchmark_burn_density' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha3' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Tue Apr 26 23:42:11 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Wed Apr 27 11:53:31 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1651027331123 ;
UFS_MODE                  (idx, 1)        = 0 ;
UFS_ORDER                 (idx, 1)        = 1.00000;
NEUTRON_TRANSPORT_MODE    (idx, 1)        = 1 ;
PHOTON_TRANSPORT_MODE     (idx, 1)        = 0 ;
GROUP_CONSTANT_GENERATION (idx, 1)        = 0 ;
B1_CALCULATION            (idx, [1:  3])  = [ 0 0 0 ];
B1_BURNUP_CORRECTION      (idx, 1)        = 0 ;

CRIT_SPEC_MODE            (idx, 1)        = 0 ;
IMPLICIT_REACTION_RATES   (idx, 1)        = 1 ;

% Optimization:

OPTIMIZATION_MODE         (idx, 1)        = 4 ;
RECONSTRUCT_MICROXS       (idx, 1)        = 1 ;
RECONSTRUCT_MACROXS       (idx, 1)        = 1 ;
DOUBLE_INDEXING           (idx, 1)        = 0 ;
MG_MAJORANT_MODE          (idx, 1)        = 0 ;
SPECTRUM_COLLAPSE         (idx, 1)        = 0 ;

% Parallelization:

MPI_TASKS                 (idx, 1)        = 3 ;
OMP_THREADS               (idx, 1)        = 18 ;
MPI_REPRODUCIBILITY       (idx, 1)        = 0 ;
OMP_REPRODUCIBILITY       (idx, 1)        = 1 ;
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.02860E+00  9.96140E-01  1.00604E+00  9.96639E-01  9.90529E-01  9.99481E-01  9.96762E-01  1.00186E+00  9.95454E-01  1.00165E+00  9.98060E-01  9.95997E-01  9.94474E-01  9.99947E-01  1.00065E+00  9.96331E-01  9.98232E-01  1.00316E+00  ];
SHARE_BUF_ARRAY           (idx, 1)        = 0 ;
SHARE_RES2_ARRAY          (idx, 1)        = 1 ;
OMP_SHARED_QUEUE_LIM      (idx, 1)        = 0 ;

% File paths:

XS_DATA_FILE_PATH         (idx, [1: 34])  = '/gv/apps/xsdata/sss_endfb7u.xsdata' ;
DECAY_DATA_FILE_PATH      (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.dec' ;
SFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
NFY_DATA_FILE_PATH        (idx, [1: 30])  = '/gv/apps/xsdata/sss_endfb7.nfy' ;
BRA_DATA_FILE_PATH        (idx, [1:  3])  = 'N/A' ;

% Collision and reaction sampling (neutrons/photons):

MIN_MACROXS               (idx, [1:   4]) = [  5.00000E-02 1.1E-09  0.00000E+00 0.0E+00 ];
DT_THRESH                 (idx, [1:  2])  = [  9.00000E-01  9.00000E-01 ];
ST_FRAC                   (idx, [1:   4]) = [  3.68540E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.63146E-01 1.3E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.25687E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.40917E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.76586E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52288E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51279E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.72448E+01 0.00033  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  2.05662E+00 0.00048  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333723 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33351E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33351E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.80772E+03 ;
RUNNING_TIME              (idx, 1)        =  7.31347E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  1.03138E+00  1.03138E+00 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.20732E+00  1.58183E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.27510E+02  2.99477E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.63867E-01  1.92333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.91333E-02  1.43167E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.28924E+01  1.97782E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.29410E+02  7.29410E+02 ];
CPU_USAGE                 (idx, 1)        = 9.30846 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98324E+00 2.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.24820E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22664.88;
MEMSIZE                   (idx, 1)        = 22480.38;
XS_MEMSIZE                (idx, 1)        = 21586.21;
MAT_MEMSIZE               (idx, 1)        = 553.39;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 328.67;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 184.50;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1114778 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 190 ;

% Nuclides and reaction channels:

TOT_NUCLIDES              (idx, 1)        = 1335 ;
TOT_TRANSPORT_NUCLIDES    (idx, 1)        = 294 ;
TOT_DOSIMETRY_NUCLIDES    (idx, 1)        = 0 ;
TOT_DECAY_NUCLIDES        (idx, 1)        = 1041 ;
TOT_PHOTON_NUCLIDES       (idx, 1)        = 0 ;
TOT_REA_CHANNELS          (idx, 1)        = 8122 ;
TOT_TRANSMU_REA           (idx, 1)        = 2355 ;

% Neutron physics options:

USE_DELNU                 (idx, 1)        = 1 ;
USE_URES                  (idx, 1)        = 1 ;
USE_DBRC                  (idx, 1)        = 0 ;
IMPL_CAPT                 (idx, 1)        = 0 ;
IMPL_NXN                  (idx, 1)        = 1 ;
IMPL_FISS                 (idx, 1)        = 0 ;
DOPPLER_PREPROCESSOR      (idx, 1)        = 1 ;
TMS_MODE                  (idx, 1)        = 0 ;
SAMPLE_FISS               (idx, 1)        = 1 ;
SAMPLE_CAPT               (idx, 1)        = 1 ;
SAMPLE_SCATT              (idx, 1)        = 1 ;

% Energy deposition:

EDEP_MODE                 (idx, 1)        = 0 ;
EDEP_DELAYED              (idx, 1)        = 1 ;
EDEP_KEFF_CORR            (idx, 1)        = 1 ;
EDEP_LOCAL_EGD            (idx, 1)        = 0 ;
EDEP_COMP                 (idx, [1:  9])  = [ 0 0 0 0 0 0 0 0 0 ];
EDEP_CAPT_E               (idx, 1)        =  0.00000E+00 ;

% Radioactivity data:

TOT_ACTIVITY              (idx, 1)        =  6.91419E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73969E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.56198E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90755E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32994E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.97196E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57216E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.58458E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.60212E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.40007E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14641E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78661E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.44422E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.03328E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33913E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31199E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52154E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.02269E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.13663E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22316E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02955E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.29724E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.99378E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  8.00506E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.64071E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.21275E-01 -4.76127E+28  2.62787E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01488E+00 0.00067 ];
TH232_FISS                (idx, [1:   4]) = [  1.52927E+18 0.00430  1.63395E-02 0.00426 ];
U233_FISS                 (idx, [1:   4]) = [  7.91362E+19 0.00055  8.45550E-01 0.00024 ];
U235_FISS                 (idx, [1:   4]) = [  7.10883E+18 0.00197  7.59573E-02 0.00194 ];
U238_FISS                 (idx, [1:   4]) = [  2.12173E+14 0.39098  2.27307E-06 0.39140 ];
PU239_FISS                (idx, [1:   4]) = [  6.57298E+17 0.00655  7.02275E-03 0.00650 ];
PU240_FISS                (idx, [1:   4]) = [  8.62920E+16 0.01846  9.22130E-04 0.01848 ];
PU241_FISS                (idx, [1:   4]) = [  1.56193E+17 0.01328  1.66888E-03 0.01327 ];
TH232_CAPT                (idx, [1:   4]) = [  9.21095E+19 0.00053  6.28452E-01 0.00037 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15154E+19 0.00155  7.85680E-02 0.00150 ];
U235_CAPT                 (idx, [1:   4]) = [  2.49235E+18 0.00320  1.70051E-02 0.00318 ];
U238_CAPT                 (idx, [1:   4]) = [  3.73184E+15 0.08292  2.54657E-05 0.08293 ];
PU239_CAPT                (idx, [1:   4]) = [  3.08953E+17 0.00928  2.10797E-03 0.00928 ];
PU240_CAPT                (idx, [1:   4]) = [  2.28822E+17 0.01076  1.56123E-03 0.01076 ];
PU241_CAPT                (idx, [1:   4]) = [  3.41143E+16 0.02918  2.32823E-04 0.02920 ];
SM149_CAPT                (idx, [1:   4]) = [  2.02393E+17 0.01161  1.38094E-03 0.01161 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10000539 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.07409E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10000539 1.00107E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5487023 5.49280E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3503999 3.50748E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 1009517 1.01046E+06 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10000539 1.00107E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -4.84288E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.61697E-02 0.0E+00  3.61697E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36688E+20 4.7E-06  2.36688E+20 4.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35890E+19 1.1E-07  9.35890E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.46588E+20 0.00029  1.28544E+20 0.00018  1.80448E+19 0.00171 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.40177E+20 0.00018  2.22133E+20 0.00011  1.80448E+19 0.00171 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.66835E+20 0.00036  2.66835E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.79051E+22 0.00026  3.58346E+22 0.00023  2.07049E+21 0.00166 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.69639E+19 0.00152 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.67141E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.20541E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.29423E+04 ;
TOT_FMASS                 (idx, 1)        =  7.35931E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.29423E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.35931E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.21371E-01 0.02950 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.06210E-01 0.00769 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.60735E-04 0.01495 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  5.31106E+03 0.03389 ];
SIX_FF_LF                 (idx, [1:   2]) = [  8.98961E-01 0.00015 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99992E-01 9.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  9.34045E-01 0.01373 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  8.39657E-01 0.01373 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52901E+00 4.6E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00072E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  8.86882E-01 0.00039  9.82179E-02 0.00040  3.42830E-04 0.00904 ];
IMP_KEFF                  (idx, [1:   2]) = [  8.86964E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  8.87052E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  8.86964E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  9.86655E-01 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.83684E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.83684E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.61076E-02 0.00147 ];
IMP_EALF                  (idx, [1:   2]) = [  1.61047E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.11394E-01 0.00145 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.11548E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  4.03756E-03 0.00518  2.75285E-04 0.01932  9.23815E-04 0.01039  7.41377E-04 0.01265  1.62900E-03 0.00840  3.79922E-04 0.01672  8.81566E-05 0.03611 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.24003E-01 0.01453  1.24775E-02 6.7E-05  3.21952E-02 0.00016  1.06756E-01 0.00053  3.01448E-01 0.00031  1.25740E+00 0.00119  6.75599E+00 0.02224 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.47845E-03 0.00787  2.40760E-04 0.02950  7.94176E-04 0.01578  6.43789E-04 0.01875  1.40346E-03 0.01240  3.20913E-04 0.02459  7.53511E-05 0.05357 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.21118E-01 0.02090  1.24775E-02 7.4E-05  3.21961E-02 0.00023  1.06743E-01 0.00070  3.01322E-01 0.00042  1.25770E+00 0.00160  7.09716E+00 0.01960 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  9.42020E-07 0.00159  9.41652E-07 0.00159  1.05051E-06 0.01779 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.35431E-07 0.00156  8.35104E-07 0.00156  9.31715E-07 0.01781 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.48645E-03 0.00917  2.38748E-04 0.03458  7.84482E-04 0.01822  6.34024E-04 0.02151  1.41349E-03 0.01472  3.31509E-04 0.02900  8.41991E-05 0.05730 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.42938E-01 0.02474  1.24783E-02 0.00011  3.21813E-02 0.00029  1.06879E-01 0.00094  3.01285E-01 0.00054  1.25381E+00 0.00214  7.25661E+00 0.02677 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  8.14765E-07 0.01985  8.14335E-07 0.01985  9.08339E-07 0.05804 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.22820E-07 0.01984  7.22439E-07 0.01984  8.06277E-07 0.05813 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.24653E-03 0.04252  2.28170E-04 0.15067  7.33680E-04 0.08572  6.13515E-04 0.09314  1.27195E-03 0.06110  3.43572E-04 0.13198  5.56475E-05 0.27181 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.01599E-01 0.08668  1.24791E-02 0.00011  3.21911E-02 0.00094  1.07110E-01 0.00327  3.01538E-01 0.00200  1.25751E+00 0.00571  8.29148E+00 0.07197 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.25050E-03 0.04160  2.30380E-04 0.15152  7.35412E-04 0.08336  6.02895E-04 0.09227  1.29281E-03 0.05993  3.33677E-04 0.13195  5.53187E-05 0.26619 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.98330E-01 0.08502  1.24791E-02 0.00011  3.21919E-02 0.00092  1.07079E-01 0.00325  3.01567E-01 0.00199  1.25817E+00 0.00564  8.29148E+00 0.07197 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.00614E+03 0.03779 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  9.29685E-07 0.00085 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.24488E-07 0.00078 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.53211E-03 0.00580 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.80054E+03 0.00593 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.49954E-09 0.00104 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.89650E-05 0.00417  6.89603E-05 0.00415  4.07073E-06 0.25139 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.41390E-05 0.01518  6.41404E-05 0.01518  3.20717E-06 0.32075 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.37998E-04 0.01464  4.38065E-04 0.01472  4.19439E-04 0.23683 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.55422E+01 0.01142 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51279E+01 0.00027  3.90467E+01 0.00039 ];

