
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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:16:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 05:44:58 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615384667 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.04448E+00  9.96289E-01  1.00710E+00  9.98019E-01  9.95412E-01  9.98200E-01  9.94492E-01  9.84660E-01  9.95403E-01  9.94859E-01  9.95574E-01  9.92816E-01  1.01908E+00  9.97612E-01  9.98327E-01  9.85836E-01  1.00272E+00  9.99121E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.98723E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70128E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40695E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54504E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61395E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.45661E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.44755E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.09835E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.24077E+00 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333948 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33401E+04 0.00064 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33401E+04 0.00064 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.60108E+02 ;
RUNNING_TIME              (idx, 1)        =  2.85577E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  9.81050E-01  9.81050E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.77917E-01  1.77917E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.73986E+01  2.73986E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.79260E+00  1.74927E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.68476E+01  0.00000E+00 ];
CPU_USAGE                 (idx, 1)        = 9.10814 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98634E+00 2.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.04528E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22056.57;
MEMSIZE                   (idx, 1)        = 21853.12;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 203.45;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  2.80441E+15 ;
TOT_DECAY_HEAT            (idx, 1)        =  2.20544E+03 ;
TOT_SF_RATE               (idx, 1)        =  3.13402E+00 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.80441E+15 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  2.20544E+03 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  0.00000E+00 ;
INHALATION_TOXICITY       (idx, 1)        =  2.69538E+10 ;
INGESTION_TOXICITY        (idx, 1)        =  1.43081E+08 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.69538E+10 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.43081E+08 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  0.00000E+00 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  0.00000E+00 ;
SR90_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
TE132_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
I131_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
I132_ACTIVITY             (idx, 1)        =  0.00000E+00 ;
CS134_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
CS137_ACTIVITY            (idx, 1)        =  0.00000E+00 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  2.29682E+10 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  0.00000E+00 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.13304E+11 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.31582E+10 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.15271E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 0 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
FIMA                      (idx, [1:  3])  = [  0.00000E+00  0.00000E+00  2.20797E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.82648E-01 0.00073 ];
TH232_FISS                (idx, [1:   4]) = [  1.54441E+18 0.00373  1.64506E-02 0.00367 ];
U233_FISS                 (idx, [1:   4]) = [  9.23349E+19 0.00053  9.83549E-01 6.1E-05 ];
TH232_CAPT                (idx, [1:   4]) = [  9.33307E+19 0.00053  7.57558E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34034E+19 0.00126  1.08797E-01 0.00124 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002040 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14140E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002040 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5162424 5.16729E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3933746 3.93752E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 905870 9.06602E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002040 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 9.68575E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36454E+20 3.7E-06  2.36454E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37767E+19 1.2E-07  9.37767E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.23226E+20 0.00033  1.09854E+20 0.00024  1.33720E+19 0.00194 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.17003E+20 0.00019  2.03631E+20 0.00013  1.33720E+19 0.00194 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.38424E+20 0.00037  2.38424E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.14219E+22 0.00027  2.97889E+22 0.00025  1.63295E+21 0.00179 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.16161E+19 0.00148 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.38619E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.06155E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51102E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.26195E-01 0.08546 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.75499E-01 0.00921 ];
SIX_FF_P                  (idx, [1:   2]) = [  4.08479E-04 0.01616 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.16552E+04 0.02361 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09350E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99989E-01 1.1E-06 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.18678E-01 0.07335 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.80681E-01 0.07335 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52145E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99671E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.92937E-01 0.00043  1.09963E-01 0.00042  3.52582E-04 0.00908 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.92084E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.91778E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.92084E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.09103E+00 0.00019 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.01592E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.01673E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.34658E-02 0.00135 ];
IMP_EALF                  (idx, [1:   2]) = [  1.34533E-02 0.00098 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.49056E-01 0.00155 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.48797E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.26364E-03 0.00556  2.65757E-04 0.01973  8.01735E-04 0.01129  5.93565E-04 0.01246  1.27938E-03 0.00893  2.70851E-04 0.01886  5.23595E-05 0.04040 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.48226E-01 0.01478  1.24759E-02 2.7E-05  3.23346E-02 8.7E-05  1.06075E-01 0.00058  2.97452E-01 0.00027  1.23566E+00 0.00022  5.30445E+00 0.03801 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.29386E-03 0.00823  2.56179E-04 0.02905  8.04526E-04 0.01683  5.97259E-04 0.01917  1.30153E-03 0.01370  2.82192E-04 0.02726  5.21738E-05 0.06593 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.52795E-01 0.02319  1.24755E-02 3.9E-05  3.23365E-02 0.00013  1.06111E-01 0.00087  2.97431E-01 0.00038  1.23600E+00 0.00029  6.30191E+00 0.02983 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.17170E-07 0.00120  8.16837E-07 0.00120  9.21341E-07 0.01542 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.11347E-07 0.00109  8.11016E-07 0.00109  9.14713E-07 0.01537 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.19995E-03 0.00933  2.57840E-04 0.03136  7.98992E-04 0.01881  5.73208E-04 0.02106  1.26141E-03 0.01518  2.55316E-04 0.03135  5.31857E-05 0.06829 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.51118E-01 0.02553  1.24754E-02 5.3E-05  3.23376E-02 0.00015  1.06053E-01 0.00103  2.97400E-01 0.00045  1.23629E+00 0.00040  6.25459E+00 0.04029 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.23585E-07 0.01956  7.23362E-07 0.01956  7.70581E-07 0.04825 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.18588E-07 0.01956  7.18366E-07 0.01956  7.65248E-07 0.04825 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.80168E-03 0.04273  1.76916E-04 0.14114  6.25642E-04 0.08322  5.27303E-04 0.08668  1.20999E-03 0.06391  2.18765E-04 0.13664  4.30675E-05 0.27763 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.58477E-01 0.10223  1.24784E-02 5.6E-05  3.23335E-02 0.00051  1.05355E-01 0.00237  2.97947E-01 0.00175  1.23919E+00 0.00083  7.52697E+00 0.10891 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.79802E-03 0.04210  1.74041E-04 0.13988  6.26316E-04 0.08196  5.36183E-04 0.08652  1.21507E-03 0.06274  2.07573E-04 0.12885  3.88367E-05 0.27296 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.56113E-01 0.10288  1.24785E-02 5.0E-05  3.23323E-02 0.00050  1.05345E-01 0.00235  2.97996E-01 0.00173  1.23915E+00 0.00084  7.52697E+00 0.10891 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.88196E+03 0.03786 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.11043E-07 0.00064 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.05275E-07 0.00051 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.21053E-03 0.00559 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.95906E+03 0.00564 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.05225E-09 0.00129 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.29170E-05 0.00478  7.28928E-05 0.00480  3.91701E-06 0.25818 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.78057E-05 0.01672  7.78060E-05 0.01685  3.93954E-06 0.33301 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  4.02523E-04 0.01618  4.02340E-04 0.01624  4.61971E-04 0.25308 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.71076E+01 0.01118 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.44755E+01 0.00027  4.01349E+01 0.00035 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:16:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 06:28:57 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615384667 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01237E+00  1.00887E+00  9.97038E-01  1.00172E+00  9.96063E-01  1.00316E+00  9.99584E-01  1.00037E+00  9.92664E-01  9.97816E-01  1.00266E+00  9.99741E-01  1.00099E+00  9.97033E-01  9.96695E-01  9.91669E-01  1.00317E+00  9.98389E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.98396E-02 0.00037  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70160E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40483E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54285E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61364E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46117E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45208E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.11435E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.24077E+00 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334587 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33415E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33415E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.74638E+02 ;
RUNNING_TIME              (idx, 1)        =  7.25505E+01 ;
INIT_TIME                 (idx, [1:  2])  = [  9.81050E-01  9.81050E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.99583E-01  1.21667E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.12442E+01  4.38456E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.89500E-02  1.89500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.49900E+00  2.70635E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.98883E+01  4.27899E+02 ];
CPU_USAGE                 (idx, 1)        = 9.29888 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99012E+00 1.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.24475E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22056.57;
MEMSIZE                   (idx, 1)        = 21853.12;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 203.45;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  5.79006E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.60537E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.65962E+01 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.49391E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.05224E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.26529E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.46920E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.48499E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  2.66138E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.42182E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  4.83151E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  2.06316E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.17823E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.44271E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.35660E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.17549E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55439E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.60101E+13 ;
CS137_ACTIVITY            (idx, 1)        =  1.38167E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  4.40567E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51656E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.51717E+13 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  7.33334E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.21980E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 1 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.23369E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.50000E+01  3.50000E+01 ];
FIMA                      (idx, [1:  3])  = [ -1.08058E-04 -2.38590E+25  2.20821E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  8.94439E-01 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.56717E+18 0.00375  1.67112E-02 0.00365 ];
U233_FISS                 (idx, [1:   4]) = [  9.21775E+19 0.00049  9.82989E-01 6.2E-05 ];
U235_FISS                 (idx, [1:   4]) = [  2.38679E+14 0.31145  2.53490E-06 0.31145 ];
TH232_CAPT                (idx, [1:   4]) = [  9.43860E+19 0.00054  7.53413E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34122E+19 0.00137  1.07060E-01 0.00132 ];
U235_CAPT                 (idx, [1:   4]) = [  1.43870E+14 0.40486  1.15134E-06 0.40487 ];
SM149_CAPT                (idx, [1:   4]) = [  1.29166E+16 0.04541  1.03136E-04 0.04545 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002439 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16326E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002439 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5200649 5.20560E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3893018 3.89651E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 908772 9.09518E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002439 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.99303E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36447E+20 3.8E-06  2.36447E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37764E+19 1.3E-07  9.37764E+19 1.3E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.25248E+20 0.00030  1.11830E+20 0.00022  1.34189E+19 0.00180 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.19025E+20 0.00017  2.05606E+20 0.00012  1.34189E+19 0.00180 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.40660E+20 0.00035  2.40660E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.17517E+22 0.00024  3.01134E+22 0.00022  1.63828E+21 0.00172 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.18891E+19 0.00140 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.40914E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.07260E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51193E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51193E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  1.63468E-01 0.08420 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.84532E-01 0.00984 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.54777E-04 0.01751 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.11906E+04 0.02577 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09055E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99993E-01 9.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  4.36002E-01 0.07034 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  3.96410E-01 0.07034 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52139E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99672E+02 1.3E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.82316E-01 0.00044  1.08811E-01 0.00041  3.49427E-04 0.00866 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.82608E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.82528E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.82608E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.08093E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.01623E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.01787E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.34614E-02 0.00131 ];
IMP_EALF                  (idx, [1:   2]) = [  1.34376E-02 0.00089 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.49365E-01 0.00147 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.49312E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.27823E-03 0.00578  2.70385E-04 0.01826  8.08620E-04 0.01212  5.97702E-04 0.01238  1.29048E-03 0.00911  2.64329E-04 0.01896  4.67123E-05 0.04740 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.37027E-01 0.01497  1.24335E-02 0.00334  3.23353E-02 9.3E-05  1.06030E-01 0.00056  2.97416E-01 0.00024  1.23523E+00 0.00022  5.13088E+00 0.04084 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.25365E-03 0.00843  2.60145E-04 0.02817  8.10396E-04 0.01681  5.87676E-04 0.01910  1.29214E-03 0.01354  2.58776E-04 0.02835  4.45119E-05 0.06535 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.32841E-01 0.01979  1.24750E-02 4.3E-05  3.23382E-02 0.00015  1.06064E-01 0.00081  2.97430E-01 0.00034  1.23505E+00 0.00029  6.60561E+00 0.02977 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.25392E-07 0.00117  8.25029E-07 0.00117  9.41421E-07 0.01636 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.10748E-07 0.00109  8.10392E-07 0.00109  9.24684E-07 0.01633 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.19613E-03 0.00895  2.66151E-04 0.03128  8.05515E-04 0.01830  5.86968E-04 0.02043  1.22853E-03 0.01497  2.62809E-04 0.02946  4.61626E-05 0.07316 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.40234E-01 0.02476  1.24747E-02 6.2E-05  3.23381E-02 0.00015  1.05942E-01 0.00090  2.97387E-01 0.00045  1.23528E+00 0.00038  6.72244E+00 0.04117 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.35722E-07 0.01956  7.35316E-07 0.01956  8.55985E-07 0.05887 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.22838E-07 0.01956  7.22439E-07 0.01956  8.40871E-07 0.05880 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.70013E-03 0.04575  2.34684E-04 0.13822  6.58732E-04 0.08578  4.32001E-04 0.09700  1.10774E-03 0.06741  2.31346E-04 0.13348  3.56294E-05 0.31807 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.17654E-01 0.07797  1.24766E-02 0.00012  3.23361E-02 0.00053  1.05622E-01 0.00282  2.96912E-01 0.00146  1.23622E+00 0.00110  5.98626E+00 0.13694 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.70906E-03 0.04582  2.28223E-04 0.13630  6.65960E-04 0.08442  4.34510E-04 0.09532  1.11084E-03 0.06768  2.31349E-04 0.13120  3.81824E-05 0.32166 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.17948E-01 0.07922  1.24766E-02 0.00012  3.23344E-02 0.00051  1.05645E-01 0.00284  2.96867E-01 0.00144  1.23622E+00 0.00109  5.98626E+00 0.13694 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.69509E+03 0.04218 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.19870E-07 0.00061 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.05328E-07 0.00045 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.13784E-03 0.00612 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.82742E+03 0.00612 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.00267E-09 0.00117 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  7.18802E-05 0.00502  7.18878E-05 0.00502  2.88847E-06 0.28719 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  7.72794E-05 0.01838  7.72594E-05 0.01843  3.53869E-06 0.38858 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.51023E-04 0.01655  3.50871E-04 0.01649  3.94498E-04 0.27273 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73497E+01 0.01171 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45208E+01 0.00026  4.01491E+01 0.00036 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:16:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 07:15:50 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615384667 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01583E+00  9.99262E-01  9.94269E-01  1.00366E+00  1.00692E+00  1.00016E+00  9.96494E-01  9.95587E-01  1.00235E+00  9.98915E-01  1.00124E+00  9.95572E-01  9.95479E-01  9.96935E-01  9.98253E-01  9.97440E-01  1.00378E+00  9.97851E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.96792E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70321E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40774E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54499E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61003E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47555E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46652E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.13306E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.23648E+00 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333698 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33380E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33380E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.11600E+03 ;
RUNNING_TIME              (idx, 1)        =  1.19439E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.81050E-01  9.81050E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  4.21333E-01  1.21750E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.17984E+02  4.67393E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  3.73667E-02  1.84167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  7.43740E+00  2.93835E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.16540E+02  6.44369E+02 ];
CPU_USAGE                 (idx, 1)        = 9.34374 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99055E+00 1.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.29108E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22056.57;
MEMSIZE                   (idx, 1)        = 21853.12;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 203.45;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.41505E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.65714E+08 ;
TOT_SF_RATE               (idx, 1)        =  4.23997E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.88309E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.31604E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.50081E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.49432E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  7.80858E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.41163E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.90934E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.08864E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  3.89901E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.60276E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.49728E+16 ;
TE132_ACTIVITY            (idx, 1)        =  4.34918E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33596E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54649E+18 ;
CS134_ACTIVITY            (idx, 1)        =  7.70638E+13 ;
CS137_ACTIVITY            (idx, 1)        =  7.18104E+16 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.03052E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.51826E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.33526E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.56674E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.30839E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 2 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.45046E+00  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.83000E+02  1.48000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.73482E-04 -1.26623E+26  2.20924E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.10057E-01 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.57223E+18 0.00394  1.67583E-02 0.00396 ];
U233_FISS                 (idx, [1:   4]) = [  9.21256E+19 0.00053  9.81899E-01 6.9E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.03297E+16 0.04992  1.10130E-04 0.04994 ];
TH232_CAPT                (idx, [1:   4]) = [  9.57376E+19 0.00054  7.47699E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34117E+19 0.00141  1.04744E-01 0.00135 ];
U235_CAPT                 (idx, [1:   4]) = [  3.72615E+15 0.08191  2.90838E-05 0.08186 ];
SM149_CAPT                (idx, [1:   4]) = [  5.79648E+16 0.02064  4.52667E-04 0.02064 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001410 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.16003E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001410 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5250521 5.25601E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3847466 3.85136E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 903423 9.04227E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001410 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.35973E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36437E+20 3.7E-06  2.36437E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37759E+19 1.1E-07  9.37759E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.28007E+20 0.00030  1.14346E+20 0.00023  1.36604E+19 0.00178 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.21782E+20 0.00017  2.08122E+20 0.00012  1.36604E+19 0.00178 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.43613E+20 0.00037  2.43613E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.21865E+22 0.00027  3.05168E+22 0.00025  1.66968E+21 0.00180 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.20282E+19 0.00128 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.43811E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.08929E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51428E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51428E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  4.59015E-01 0.06763 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.03275E-01 0.01162 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.22955E-04 0.02247 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.37758E+03 0.03036 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09580E-01 0.00012 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99996E-01 6.3E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.11833E-01 0.04989 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.56481E-01 0.04989 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52130E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99673E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.71013E-01 0.00042  1.07549E-01 0.00040  3.46511E-04 0.00882 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.70891E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.70584E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.70891E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.06744E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02317E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02329E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33687E-02 0.00135 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33654E-02 0.00099 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.50450E-01 0.00156 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.50247E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.34153E-03 0.00561  2.70568E-04 0.02011  8.22350E-04 0.01210  6.06098E-04 0.01317  1.31492E-03 0.00831  2.78362E-04 0.01834  4.92387E-05 0.04918 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.43424E-01 0.01611  1.24753E-02 3.1E-05  3.23366E-02 9.4E-05  1.06174E-01 0.00063  2.97384E-01 0.00028  1.23597E+00 0.00024  4.85874E+00 0.04284 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.23824E-03 0.00761  2.55991E-04 0.02990  7.84094E-04 0.01664  6.05615E-04 0.01846  1.27833E-03 0.01277  2.68975E-04 0.02994  4.52362E-05 0.06616 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.38226E-01 0.02072  1.24759E-02 3.5E-05  3.23457E-02 0.00015  1.06139E-01 0.00089  2.97379E-01 0.00040  1.23627E+00 0.00030  6.28830E+00 0.03094 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.38147E-07 0.00119  8.37817E-07 0.00119  9.44153E-07 0.01636 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.13805E-07 0.00110  8.13485E-07 0.00110  9.16694E-07 0.01635 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.21561E-03 0.00886  2.51191E-04 0.03170  7.90347E-04 0.01960  5.74874E-04 0.01940  1.28591E-03 0.01430  2.70030E-04 0.03071  4.32557E-05 0.07819 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.34174E-01 0.02334  1.24756E-02 5.2E-05  3.23350E-02 0.00016  1.06248E-01 0.00106  2.97195E-01 0.00043  1.23591E+00 0.00039  6.27981E+00 0.04582 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.43040E-07 0.01977  7.42723E-07 0.01978  8.42169E-07 0.05986 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.21742E-07 0.01977  7.21433E-07 0.01977  8.18366E-07 0.05995 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.73629E-03 0.04202  1.94196E-04 0.14209  6.55832E-04 0.08472  4.81260E-04 0.08801  1.08296E-03 0.06291  2.76358E-04 0.12553  4.56782E-05 0.37534 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.42034E-01 0.07922  1.24756E-02 0.00015  3.23447E-02 0.00057  1.06697E-01 0.00380  2.97931E-01 0.00183  1.23600E+00 0.00104  6.26138E+00 0.15198 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.76067E-03 0.04169  1.95793E-04 0.13861  6.60762E-04 0.08123  4.77699E-04 0.08766  1.10119E-03 0.06204  2.78152E-04 0.12291  4.70784E-05 0.38101 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.46049E-01 0.08438  1.24756E-02 0.00015  3.23454E-02 0.00057  1.06740E-01 0.00380  2.97947E-01 0.00183  1.23601E+00 0.00104  6.26138E+00 0.15198 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.69473E+03 0.03740 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.33117E-07 0.00066 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.08922E-07 0.00049 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.23938E-03 0.00633 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.88866E+03 0.00634 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.84210E-09 0.00093 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.94764E-05 0.00654  6.94827E-05 0.00650  6.84472E-07 0.65530 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.98275E-05 0.02656  6.98236E-05 0.02660  3.90029E-07 0.63552 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.21063E-04 0.02140  2.21500E-04 0.02142  9.03201E-05 0.57580 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.72770E+01 0.01138 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46652E+01 0.00026  4.03233E+01 0.00036 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:16:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 08:03:19 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615384667 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01119E+00  9.98091E-01  1.00953E+00  9.94603E-01  9.97602E-01  1.00308E+00  1.00163E+00  9.94559E-01  9.99816E-01  1.00603E+00  9.94515E-01  9.96720E-01  9.99968E-01  9.94804E-01  9.97391E-01  9.97224E-01  1.00515E+00  9.98101E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.95983E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70402E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40741E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54439E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60489E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48296E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47389E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.14861E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.23355E+00 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334183 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33388E+04 0.00064 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33388E+04 0.00064 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  1.56351E+03 ;
RUNNING_TIME              (idx, 1)        =  1.66912E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.81050E-01  9.81050E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  5.46233E-01  1.24900E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  1.65300E+02  4.73167E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  5.73667E-02  2.00000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.03373E+01  2.89980E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  1.64051E+02  6.79778E+02 ];
CPU_USAGE                 (idx, 1)        = 9.36727 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99063E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.31582E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22056.57;
MEMSIZE                   (idx, 1)        = 21853.12;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 203.45;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.53203E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67339E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.15480E+02 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91886E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.34011E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.58164E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50779E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  8.86210E+11 ;
INGESTION_TOXICITY        (idx, 1)        =  3.57345E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.06228E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.27568E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  4.79872E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.74586E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.48894E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.36734E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34993E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56527E+18 ;
CS134_ACTIVITY            (idx, 1)        =  1.40076E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.42644E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.10359E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.54872E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  4.17124E+14 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.72574E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.36287E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 3 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28657E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+02  1.82000E+02 ];
FIMA                      (idx, [1:  3])  = [ -1.12055E-03 -2.47415E+26  2.21045E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.22867E-01 0.00076 ];
TH232_FISS                (idx, [1:   4]) = [  1.58536E+18 0.00383  1.69139E-02 0.00374 ];
U233_FISS                 (idx, [1:   4]) = [  9.18777E+19 0.00054  9.80280E-01 7.2E-05 ];
U235_FISS                 (idx, [1:   4]) = [  4.17808E+16 0.02454  4.45568E-04 0.02447 ];
TH232_CAPT                (idx, [1:   4]) = [  9.65888E+19 0.00053  7.44654E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.34205E+19 0.00133  1.03467E-01 0.00129 ];
U235_CAPT                 (idx, [1:   4]) = [  1.54326E+16 0.04150  1.18979E-04 0.04150 ];
SM149_CAPT                (idx, [1:   4]) = [  9.79739E+16 0.01686  7.55338E-04 0.01686 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001629 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15586E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001629 1.00116E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5279605 5.28505E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3815212 3.81889E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 906812 9.07619E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001629 1.00116E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -3.33413E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36431E+20 3.5E-06  2.36431E+20 3.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37751E+19 1.2E-07  9.37751E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.29654E+20 0.00031  1.15910E+20 0.00022  1.37442E+19 0.00195 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.23429E+20 0.00018  2.09685E+20 0.00012  1.37442E+19 0.00195 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.45429E+20 0.00035  2.45429E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.24541E+22 0.00026  3.07807E+22 0.00023  1.67337E+21 0.00190 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.22762E+19 0.00146 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.45705E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.09923E+22 0.00035 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51429E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51429E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.50782E-01 0.05861 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.93296E-01 0.01436 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.79757E-04 0.02574 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  1.00326E+04 0.02675 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09241E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.63435E-01 0.04464 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.03287E-01 0.04464 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52125E+00 3.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99675E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.62770E-01 0.00046  1.06636E-01 0.00045  3.45106E-04 0.00953 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.63379E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.63373E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.63379E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.05955E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02525E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02506E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33408E-02 0.00136 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33414E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.51194E-01 0.00164 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.51235E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.38143E-03 0.00587  2.72370E-04 0.01973  8.54789E-04 0.01082  6.13670E-04 0.01378  1.30467E-03 0.00916  2.81376E-04 0.01953  5.45613E-05 0.04641 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.51271E-01 0.01584  1.24760E-02 2.8E-05  3.23269E-02 8.6E-05  1.06015E-01 0.00058  2.97377E-01 0.00027  1.23559E+00 0.00026  5.14482E+00 0.03969 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.25686E-03 0.00840  2.57331E-04 0.02905  8.23363E-04 0.01543  5.77341E-04 0.02011  1.28103E-03 0.01355  2.68826E-04 0.02992  4.89723E-05 0.06708 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.45234E-01 0.02189  1.24755E-02 4.4E-05  3.23251E-02 0.00012  1.06037E-01 0.00077  2.97255E-01 0.00037  1.23580E+00 0.00038  6.43947E+00 0.02935 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.46453E-07 0.00127  8.46143E-07 0.00127  9.43106E-07 0.01627 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.14878E-07 0.00115  8.14581E-07 0.00115  9.07733E-07 0.01621 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.21758E-03 0.00977  2.47711E-04 0.03418  8.16506E-04 0.01799  5.85330E-04 0.02189  1.24300E-03 0.01518  2.72494E-04 0.03416  5.25314E-05 0.07811 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.55242E-01 0.02651  1.24756E-02 5.6E-05  3.23239E-02 0.00016  1.05878E-01 0.00087  2.97366E-01 0.00045  1.23587E+00 0.00042  6.60260E+00 0.03967 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.48892E-07 0.01957  7.48611E-07 0.01957  8.27656E-07 0.05674 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.21054E-07 0.01956  7.20784E-07 0.01956  7.96900E-07 0.05675 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.07733E-03 0.04402  2.35510E-04 0.12244  7.74047E-04 0.07795  5.74002E-04 0.08894  1.20228E-03 0.06516  2.36232E-04 0.14582  5.52581E-05 0.24714 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.25562E-01 0.08687  1.24776E-02 9.8E-05  3.23204E-02 0.00047  1.05612E-01 0.00255  2.97437E-01 0.00154  1.23741E+00 0.00102  5.60108E+00 0.12167 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.06395E-03 0.04256  2.37385E-04 0.12042  7.76211E-04 0.07610  5.68940E-04 0.08717  1.19208E-03 0.06317  2.33777E-04 0.14061  5.55562E-05 0.25034 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.17106E-01 0.08044  1.24775E-02 9.8E-05  3.23208E-02 0.00047  1.05603E-01 0.00254  2.97476E-01 0.00155  1.23744E+00 0.00102  5.60108E+00 0.12167 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.12311E+03 0.03938 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.39746E-07 0.00068 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.08431E-07 0.00050 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.24442E-03 0.00568 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.86367E+03 0.00564 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.78654E-09 0.00082 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.75548E-05 0.00749  6.75567E-05 0.00752  8.71525E-07 0.50526 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.15136E-05 0.02663  6.14544E-05 0.02665  1.08246E-06 0.63152 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.78043E-04 0.02447  1.78243E-04 0.02451  1.14590E-04 0.49954 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.75111E+01 0.01126 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47389E+01 0.00027  4.04047E+01 0.00036 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:16:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 08:52:35 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615384667 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01224E+00  9.98375E-01  1.00035E+00  1.00610E+00  1.00157E+00  9.99830E-01  9.87880E-01  9.98698E-01  9.98169E-01  1.00584E+00  9.90751E-01  1.00341E+00  9.95817E-01  9.95273E-01  9.98188E-01  1.00050E+00  1.00469E+00  1.00231E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.95565E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70443E-01 9.6E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40755E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54438E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60191E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49589E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48681E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.17218E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.23454E+00 0.00048  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334063 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33401E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33401E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.02540E+03 ;
RUNNING_TIME              (idx, 1)        =  2.16180E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.81050E-01  9.81050E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  6.83567E-01  1.37333E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.14396E+02  4.90961E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  7.68000E-02  1.94333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.36158E+01  3.27852E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.12946E+02  6.87682E+02 ];
CPU_USAGE                 (idx, 1)        = 9.36907 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99029E+00 1.6E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.31749E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22056.57;
MEMSIZE                   (idx, 1)        = 21853.12;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 203.45;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.56028E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.66933E+08 ;
TOT_SF_RATE               (idx, 1)        =  2.50592E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.93775E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35167E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.59082E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50241E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.00428E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.69555E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.32924E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.36182E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  5.70787E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.85926E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.93912E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.33615E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.32606E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.53232E+18 ;
CS134_ACTIVITY            (idx, 1)        =  2.35790E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.81709E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.10227E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.52640E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.51198E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.76993E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.45670E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 4 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57313E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30000E+02  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -2.26514E-03 -5.00136E+26  2.21297E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.44831E-01 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.61207E+18 0.00399  1.71914E-02 0.00397 ];
U233_FISS                 (idx, [1:   4]) = [  9.15841E+19 0.00052  9.76670E-01 7.9E-05 ];
U235_FISS                 (idx, [1:   4]) = [  1.61171E+17 0.01145  1.71902E-03 0.01149 ];
TH232_CAPT                (idx, [1:   4]) = [  9.80968E+19 0.00052  7.40464E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.33641E+19 0.00133  1.00878E-01 0.00130 ];
U235_CAPT                 (idx, [1:   4]) = [  5.48904E+16 0.02072  4.14345E-04 0.02073 ];
SM149_CAPT                (idx, [1:   4]) = [  1.35163E+17 0.01360  1.02029E-03 0.01362 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002034 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.15182E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002034 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5324798 5.33001E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3769230 3.77268E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 908006 9.08829E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002034 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -1.39698E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36420E+20 3.7E-06  2.36420E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37730E+19 1.1E-07  9.37730E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.32416E+20 0.00030  1.18469E+20 0.00022  1.39470E+19 0.00184 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.26189E+20 0.00017  2.12242E+20 0.00012  1.39470E+19 0.00184 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.48557E+20 0.00035  2.48557E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.29204E+22 0.00024  3.12275E+22 0.00022  1.69295E+21 0.00188 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.25906E+19 0.00153 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.48780E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.11645E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.51149E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.51149E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.63521E-01 0.05577 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.27500E-01 0.01314 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.61062E-04 0.02584 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.71089E+03 0.02643 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09119E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.1E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.97341E-01 0.04090 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.33992E-01 0.04090 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52120E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99679E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.51094E-01 0.00044  1.05342E-01 0.00044  3.42109E-04 0.00907 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.51427E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.51208E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.51427E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.04657E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02237E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02184E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33791E-02 0.00132 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33845E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.53832E-01 0.00153 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.53647E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.44378E-03 0.00556  2.73349E-04 0.02201  8.54568E-04 0.01032  6.37820E-04 0.01303  1.34315E-03 0.00899  2.81827E-04 0.01856  5.30684E-05 0.04614 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.43367E-01 0.01598  1.24760E-02 2.7E-05  3.23306E-02 9.2E-05  1.06200E-01 0.00062  2.97550E-01 0.00027  1.23579E+00 0.00026  4.95869E+00 0.04054 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.25540E-03 0.00813  2.53911E-04 0.03206  8.05257E-04 0.01604  6.12298E-04 0.01974  1.26982E-03 0.01327  2.64889E-04 0.02699  4.92190E-05 0.06743 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.45882E-01 0.02432  1.24755E-02 4.1E-05  3.23238E-02 0.00012  1.06147E-01 0.00077  2.97616E-01 0.00040  1.23545E+00 0.00035  6.30252E+00 0.03053 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.56853E-07 0.00125  8.56512E-07 0.00125  9.66249E-07 0.01528 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.14891E-07 0.00113  8.14566E-07 0.00114  9.19136E-07 0.01533 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.23846E-03 0.00926  2.54073E-04 0.03437  8.21026E-04 0.01748  5.93449E-04 0.02156  1.25553E-03 0.01514  2.65759E-04 0.03107  4.86180E-05 0.07949 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.35800E-01 0.02560  1.24768E-02 3.7E-05  3.23374E-02 0.00016  1.06163E-01 0.00100  2.97564E-01 0.00046  1.23614E+00 0.00044  5.99317E+00 0.04535 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.57742E-07 0.01965  7.57534E-07 0.01966  8.18942E-07 0.05394 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.20993E-07 0.01965  7.20795E-07 0.01965  7.79449E-07 0.05396 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.88838E-03 0.04403  2.17629E-04 0.13829  8.12095E-04 0.06941  5.61579E-04 0.09164  1.06990E-03 0.06927  1.95298E-04 0.14674  3.18782E-05 0.29991 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  2.92777E-01 0.10175  1.24745E-02 0.00016  3.23472E-02 0.00059  1.06016E-01 0.00308  2.97702E-01 0.00181  1.23943E+00 0.00157  5.08278E+00 0.13735 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.89676E-03 0.04356  2.23441E-04 0.13838  8.10294E-04 0.06800  5.52129E-04 0.09083  1.08209E-03 0.06833  1.98873E-04 0.14452  2.99276E-05 0.29306 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  2.86531E-01 0.09036  1.24745E-02 0.00016  3.23483E-02 0.00058  1.06061E-01 0.00314  2.97716E-01 0.00182  1.23951E+00 0.00159  5.02905E+00 0.13798 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.83181E+03 0.04029 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.51368E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.09686E-07 0.00058 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.22160E-03 0.00649 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.78400E+03 0.00645 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.77309E-09 0.00077 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.66814E-05 0.00867  6.66820E-05 0.00867  4.31688E-07 0.71706 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.19042E-05 0.02748  6.17200E-05 0.02761  7.47264E-07 0.92206 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.58827E-04 0.02433  1.59178E-04 0.02435  6.62564E-05 0.70592 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.74447E+01 0.01151 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48681E+01 0.00028  4.04579E+01 0.00036 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:16:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 09:42:08 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615384667 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01950E+00  1.00270E+00  1.00450E+00  1.00477E+00  9.94321E-01  1.00087E+00  9.98847E-01  1.00065E+00  1.00208E+00  9.96990E-01  9.91729E-01  9.95526E-01  1.00265E+00  1.00241E+00  9.91347E-01  9.93057E-01  1.00107E+00  9.97000E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.95177E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70482E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40558E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.54232E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60109E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.50686E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.49779E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.19954E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.23538E+00 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334368 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33439E+04 0.00057 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33439E+04 0.00057 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.47641E+03 ;
RUNNING_TIME              (idx, 1)        =  2.65727E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.81050E-01  9.81050E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  8.12833E-01  1.29267E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  2.63777E+02  4.93805E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  9.66000E-02  1.98000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  1.84175E+01  4.80163E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  2.60964E+02  7.04171E+02 ];
CPU_USAGE                 (idx, 1)        = 9.31940 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99053E+00 1.4E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.26187E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22056.57;
MEMSIZE                   (idx, 1)        = 21853.12;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 203.45;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.62473E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67841E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.12175E+03 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.96997E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37282E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.62266E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50901E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.10117E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.80447E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  4.73290E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.51083E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.26402E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  2.95309E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.35591E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.34568E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33313E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.54183E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.03449E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.17713E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.13892E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.55718E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  3.33913E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.86438E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.52462E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 5 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.85970E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09500E+03  3.65000E+02 ];
FIMA                      (idx, [1:  3])  = [ -3.39378E-03 -7.49338E+26  2.21546E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.61835E-01 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.60988E+18 0.00440  1.71585E-02 0.00432 ];
U233_FISS                 (idx, [1:   4]) = [  9.12590E+19 0.00048  9.72743E-01 9.4E-05 ];
U235_FISS                 (idx, [1:   4]) = [  3.35699E+17 0.00900  3.57849E-03 0.00901 ];
TH232_CAPT                (idx, [1:   4]) = [  9.91064E+19 0.00052  7.36778E-01 0.00028 ];
U233_CAPT                 (idx, [1:   4]) = [  1.32686E+19 0.00135  9.86442E-02 0.00134 ];
U235_CAPT                 (idx, [1:   4]) = [  1.16381E+17 0.01469  8.65095E-04 0.01465 ];
SM149_CAPT                (idx, [1:   4]) = [  1.53575E+17 0.01220  1.14175E-03 0.01222 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10003170 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14476E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10003170 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5358282 5.36293E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3737404 3.74040E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 907484 9.08123E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10003170 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.57045E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36407E+20 3.7E-06  2.36407E+20 3.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37701E+19 1.1E-07  9.37701E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.34558E+20 0.00029  1.20459E+20 0.00021  1.40991E+19 0.00190 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.28328E+20 0.00017  2.14229E+20 0.00012  1.40991E+19 0.00190 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.50821E+20 0.00034  2.50821E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.32860E+22 0.00025  3.15741E+22 0.00023  1.71190E+21 0.00182 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.27783E+19 0.00146 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.51107E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.12937E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.50744E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.50744E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.37112E-01 0.05308 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.25637E-01 0.01299 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.60720E-04 0.02673 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.14832E+03 0.02855 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09190E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.5E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.98196E-01 0.04028 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.34846E-01 0.04028 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52113E+00 3.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99685E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.42890E-01 0.00040  1.04447E-01 0.00039  3.33626E-04 0.00973 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.42554E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.42565E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.42554E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.03670E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.02257E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  7.02099E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.33767E-02 0.00138 ];
IMP_EALF                  (idx, [1:   2]) = [  1.33958E-02 0.00091 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.56173E-01 0.00162 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.55937E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.46488E-03 0.00530  2.78822E-04 0.01854  8.51235E-04 0.01071  6.40912E-04 0.01237  1.35563E-03 0.00887  2.83871E-04 0.02055  5.44014E-05 0.04296 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.47056E-01 0.01539  1.24757E-02 2.8E-05  3.23282E-02 0.00010  1.06221E-01 0.00061  2.97655E-01 0.00027  1.23727E+00 0.00035  5.17810E+00 0.03803 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.22313E-03 0.00851  2.53189E-04 0.02838  7.87944E-04 0.01637  5.98391E-04 0.01799  1.28206E-03 0.01360  2.51667E-04 0.02788  4.98825E-05 0.06339 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.42219E-01 0.02220  1.24758E-02 3.6E-05  3.23286E-02 0.00014  1.06282E-01 0.00085  2.97521E-01 0.00039  1.23751E+00 0.00054  6.26174E+00 0.02938 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.67720E-07 0.00127  8.67308E-07 0.00128  9.85847E-07 0.01565 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.18126E-07 0.00121  8.17737E-07 0.00122  9.29572E-07 0.01568 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.18999E-03 0.00989  2.44633E-04 0.03590  7.86676E-04 0.01874  5.96646E-04 0.02093  1.24962E-03 0.01557  2.62908E-04 0.03302  4.95118E-05 0.07442 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.50582E-01 0.02473  1.24762E-02 5.7E-05  3.23252E-02 0.00016  1.06207E-01 0.00104  2.97556E-01 0.00045  1.23778E+00 0.00066  6.70526E+00 0.04062 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.70903E-07 0.01968  7.70547E-07 0.01968  8.63148E-07 0.06176 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.26973E-07 0.01968  7.26639E-07 0.01968  8.13325E-07 0.06159 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.84127E-03 0.04598  2.38359E-04 0.14657  7.27011E-04 0.08078  5.64228E-04 0.08982  1.04418E-03 0.06815  2.27080E-04 0.13175  4.04147E-05 0.31433 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.17521E-01 0.08382  1.24762E-02 0.00014  3.23264E-02 0.00059  1.06087E-01 0.00341  2.98233E-01 0.00192  1.23631E+00 0.00172  6.32329E+00 0.14044 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.83926E-03 0.04606  2.45316E-04 0.14161  7.15905E-04 0.07984  5.65485E-04 0.08970  1.04741E-03 0.06801  2.25436E-04 0.13037  3.97098E-05 0.30848 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.14032E-01 0.07880  1.24761E-02 0.00014  3.23281E-02 0.00060  1.06087E-01 0.00340  2.98089E-01 0.00187  1.23612E+00 0.00170  6.32329E+00 0.14044 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.70141E+03 0.04196 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.61963E-07 0.00073 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.12694E-07 0.00058 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.17785E-03 0.00641 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.68812E+03 0.00656 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.78453E-09 0.00079 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.71613E-05 0.00868  6.71317E-05 0.00864  1.00327E-06 0.54830 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.09388E-05 0.03030  6.10088E-05 0.03034  6.63863E-07 0.54094 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.58614E-04 0.02550  1.58761E-04 0.02539  1.01784E-04 0.49797 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.73255E+01 0.01096 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.49779E+01 0.00027  4.05203E+01 0.00036 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:16:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 10:32:44 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615384667 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01437E+00  1.00231E+00  1.00119E+00  1.00431E+00  1.00140E+00  1.00079E+00  1.00566E+00  9.92164E-01  1.00131E+00  9.90238E-01  9.94937E-01  9.96505E-01  1.00250E+00  1.00068E+00  9.98289E-01  1.00458E+00  9.97784E-01  9.90968E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.95318E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70468E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40151E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53838E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59947E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.51432E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.50521E+01 0.00025  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.22726E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.23979E+00 0.00055  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334085 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33406E+04 0.00062 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33406E+04 0.00062 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  2.93672E+03 ;
RUNNING_TIME              (idx, 1)        =  3.16331E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.81050E-01  9.81050E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  9.45667E-01  1.32833E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.14208E+02  5.04316E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.16117E-01  1.95167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.33615E+01  4.94395E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.11427E+02  7.07690E+02 ];
CPU_USAGE                 (idx, 1)        = 9.28370 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99103E+00 1.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.22175E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22056.57;
MEMSIZE                   (idx, 1)        = 21853.12;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 203.45;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 176 ;

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

TOT_ACTIVITY              (idx, 1)        =  6.65252E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.67738E+08 ;
TOT_SF_RATE               (idx, 1)        =  5.10209E+04 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.99322E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38628E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.62683E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.50629E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.27246E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  3.93925E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  5.72554E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.64695E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  6.95335E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.07365E+11 ;
SR90_ACTIVITY             (idx, 1)        =  7.07842E+17 ;
TE132_ACTIVITY            (idx, 1)        =  4.32214E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31425E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51622E+18 ;
CS134_ACTIVITY            (idx, 1)        =  3.86013E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.79495E+17 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.14412E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.56895E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  8.91022E+15 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.90572E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.62038E+15 0.00036  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 6 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43283E+01  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82500E+03  7.30000E+02 ];
FIMA                      (idx, [1:  3])  = [ -5.72633E-03 -1.26436E+27  2.22062E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  9.87746E-01 0.00075 ];
TH232_FISS                (idx, [1:   4]) = [  1.62286E+18 0.00388  1.73032E-02 0.00383 ];
U233_FISS                 (idx, [1:   4]) = [  9.03871E+19 0.00052  9.63747E-01 1.0E-04 ];
U235_FISS                 (idx, [1:   4]) = [  8.11589E+17 0.00523  8.65376E-03 0.00522 ];
TH232_CAPT                (idx, [1:   4]) = [  1.00445E+20 0.00053  7.31286E-01 0.00027 ];
U233_CAPT                 (idx, [1:   4]) = [  1.31714E+19 0.00138  9.58949E-02 0.00132 ];
U235_CAPT                 (idx, [1:   4]) = [  2.91667E+17 0.00937  2.12354E-03 0.00937 ];
SM149_CAPT                (idx, [1:   4]) = [  1.61434E+17 0.01269  1.17551E-03 0.01273 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002182 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14229E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002182 1.00114E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5402265 5.40739E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3688886 3.69225E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 911031 9.11778E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002182 1.00114E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 3.94881E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36375E+20 3.8E-06  2.36375E+20 3.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37628E+19 1.2E-07  9.37628E+19 1.2E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.37430E+20 0.00030  1.23165E+20 0.00022  1.42645E+19 0.00178 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.31193E+20 0.00018  2.16928E+20 0.00013  1.42645E+19 0.00178 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.54013E+20 0.00036  2.54013E+20 0.00036  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.37752E+22 0.00025  3.20530E+22 0.00023  1.72211E+21 0.00166 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.31614E+19 0.00153 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.54354E+20 0.00022 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.14562E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.49921E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.49921E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.17751E-01 0.05905 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.19900E-01 0.01500 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.52634E-04 0.02840 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.48128E+03 0.02715 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08824E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 4.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.31886E-01 0.04560 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.74222E-01 0.04560 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52099E+00 3.7E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99701E+02 1.2E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.30743E-01 0.00045  1.03083E-01 0.00044  3.41913E-04 0.00909 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.30398E-01 0.00022 ];
COL_KEFF                  (idx, [1:   2]) = [  9.30600E-01 0.00035 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.30398E-01 0.00022 ];
ABS_KINF                  (idx, [1:   2]) = [  1.02372E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  7.01355E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  7.01346E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.34979E-02 0.00137 ];
IMP_EALF                  (idx, [1:   2]) = [  1.34973E-02 0.00097 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.60589E-01 0.00153 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.60554E-01 0.00048 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.53193E-03 0.00550  2.77620E-04 0.01965  8.67277E-04 0.01192  6.50940E-04 0.01241  1.38511E-03 0.00858  2.95009E-04 0.01812  5.59760E-05 0.04314 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.50281E-01 0.01467  1.24755E-02 2.8E-05  3.23183E-02 0.00010  1.06139E-01 0.00058  2.97983E-01 0.00028  1.23933E+00 0.00038  5.32124E+00 0.03699 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.21919E-03 0.00827  2.51775E-04 0.02955  7.84335E-04 0.01653  5.98415E-04 0.01933  1.26361E-03 0.01287  2.69321E-04 0.02710  5.17328E-05 0.06647 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.50839E-01 0.02209  1.24749E-02 4.4E-05  3.23227E-02 0.00014  1.06088E-01 0.00075  2.98005E-01 0.00040  1.23920E+00 0.00050  6.43310E+00 0.02886 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.78368E-07 0.00131  8.78059E-07 0.00131  9.74818E-07 0.01796 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.17495E-07 0.00126  8.17208E-07 0.00126  9.07246E-07 0.01793 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.30425E-03 0.00923  2.60173E-04 0.03228  8.15133E-04 0.01787  6.08025E-04 0.02106  1.29996E-03 0.01591  2.73404E-04 0.03013  4.75521E-05 0.07287 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.37478E-01 0.02275  1.24754E-02 5.7E-05  3.23182E-02 0.00018  1.06151E-01 0.00099  2.98038E-01 0.00046  1.23886E+00 0.00056  6.33856E+00 0.04157 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.77293E-07 0.01961  7.77027E-07 0.01961  8.29895E-07 0.05363 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.23640E-07 0.01961  7.23392E-07 0.01961  7.72348E-07 0.05356 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.98876E-03 0.04480  2.67602E-04 0.13630  7.55717E-04 0.08589  5.37039E-04 0.09425  1.17002E-03 0.06084  2.11493E-04 0.15291  4.68929E-05 0.30795 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.14482E-01 0.08631  1.24733E-02 0.00017  3.23310E-02 0.00067  1.06094E-01 0.00320  2.98340E-01 0.00176  1.24129E+00 0.00223  6.42090E+00 0.12368 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.99695E-03 0.04462  2.64799E-04 0.13594  7.47018E-04 0.08575  5.44959E-04 0.09184  1.17921E-03 0.05971  2.16841E-04 0.15239  4.41206E-05 0.29897 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.16196E-01 0.08461  1.24734E-02 0.00017  3.23314E-02 0.00066  1.06078E-01 0.00319  2.98273E-01 0.00174  1.24134E+00 0.00222  6.42090E+00 0.12368 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.87399E+03 0.04073 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.71632E-07 0.00071 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.11211E-07 0.00051 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.30802E-03 0.00511 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.79650E+03 0.00529 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.79385E-09 0.00079 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.73120E-05 0.00988  6.72404E-05 0.00991  2.62923E-06 0.32187 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.90215E-05 0.02975  5.91564E-05 0.02986  1.43944E-06 0.41465 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.51436E-04 0.02716  1.50971E-04 0.02721  2.81662E-04 0.31187 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.70974E+01 0.01163 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.50521E+01 0.00025  4.04796E+01 0.00036 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:16:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 11:24:10 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615384667 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01341E+00  1.00606E+00  9.98160E-01  1.00174E+00  9.98449E-01  1.00962E+00  9.94500E-01  1.00149E+00  1.00321E+00  9.94583E-01  9.93295E-01  9.93780E-01  9.97371E-01  9.98464E-01  1.00265E+00  9.89081E-01  1.00261E+00  1.00153E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.95545E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70446E-01 9.5E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40018E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53714E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59465E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.52106E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.51201E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.24415E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.24334E+00 0.00050  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333592 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33392E+04 0.00060 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33392E+04 0.00060 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.40476E+03 ;
RUNNING_TIME              (idx, 1)        =  3.67758E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.81050E-01  9.81050E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.08083E+00  1.35167E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  3.65456E+02  5.12479E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.36833E-01  2.07167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  2.83546E+01  4.99303E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  3.62804E+02  7.16917E+02 ];
CPU_USAGE                 (idx, 1)        = 9.25815 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99121E+00 1.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.19309E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22056.57;
MEMSIZE                   (idx, 1)        = 21853.12;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 203.45;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 178 ;

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

TOT_ACTIVITY              (idx, 1)        =  6.72598E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.69037E+08 ;
TOT_SF_RATE               (idx, 1)        =  9.93306E+05 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01561E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39632E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.67729E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.51772E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.75308E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.23903E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  9.11651E+11 ;
ACTINIDE_ING_TOX          (idx, 1)        =  8.87090E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  8.25200E+11 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.34874E+11 ;
SR90_ACTIVITY             (idx, 1)        =  1.33167E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.32013E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30959E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51137E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.59624E+14 ;
CS137_ACTIVITY            (idx, 1)        =  1.28185E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.17734E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.67311E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.84298E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.97912E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.72784E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 7 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28657E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65000E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -1.16098E-02 -2.56342E+27  2.23361E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01882E+00 0.00074 ];
TH232_FISS                (idx, [1:   4]) = [  1.61301E+18 0.00385  1.72205E-02 0.00381 ];
U233_FISS                 (idx, [1:   4]) = [  8.80738E+19 0.00049  9.40281E-01 0.00013 ];
U235_FISS                 (idx, [1:   4]) = [  2.27688E+18 0.00316  2.43085E-02 0.00314 ];
PU239_FISS                (idx, [1:   4]) = [  7.20248E+14 0.18720  7.68953E-06 0.18706 ];
PU240_FISS                (idx, [1:   4]) = [  2.57890E+13 1.00000  2.75028E-07 1.00000 ];
TH232_CAPT                (idx, [1:   4]) = [  1.00950E+20 0.00053  7.16574E-01 0.00029 ];
U233_CAPT                 (idx, [1:   4]) = [  1.28765E+19 0.00137  9.14029E-02 0.00133 ];
U235_CAPT                 (idx, [1:   4]) = [  8.09878E+17 0.00558  5.74914E-03 0.00560 ];
U238_CAPT                 (idx, [1:   4]) = [  2.57744E+13 1.00000  1.83493E-07 1.00000 ];
PU239_CAPT                (idx, [1:   4]) = [  2.06589E+14 0.34940  1.46695E-06 0.34940 ];
SM149_CAPT                (idx, [1:   4]) = [  1.71397E+17 0.01238  1.21655E-03 0.01235 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001747 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.14699E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001747 1.00115E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5463489 5.46899E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3632879 3.63628E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 905379 9.06200E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001747 1.00115E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -9.31323E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36281E+20 4.0E-06  2.36281E+20 4.0E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37411E+19 1.1E-07  9.37411E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.40945E+20 0.00028  1.26332E+20 0.00021  1.46130E+19 0.00186 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.34686E+20 0.00017  2.20073E+20 0.00012  1.46130E+19 0.00186 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.57595E+20 0.00034  2.57595E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.43002E+22 0.00024  3.25410E+22 0.00021  1.75921E+21 0.00188 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.33443E+19 0.00152 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.58030E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16353E+22 0.00032 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.47820E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.47820E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  7.27268E-01 0.05209 ];
SIX_FF_F                  (idx, [1:   2]) = [  7.75506E-01 0.01872 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.52387E-04 0.02677 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.99407E+03 0.02771 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09381E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 4.2E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.68540E-01 0.04120 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.07918E-01 0.04120 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52057E+00 4.0E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99747E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.16361E-01 0.00044  1.01500E-01 0.00043  3.39447E-04 0.00892 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.16763E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.17291E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.16763E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00806E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.98493E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  6.98525E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.38897E-02 0.00135 ];
IMP_EALF                  (idx, [1:   2]) = [  1.38831E-02 0.00089 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.69680E-01 0.00136 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.69931E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.67618E-03 0.00506  2.98507E-04 0.01938  8.82747E-04 0.01130  6.68667E-04 0.01272  1.44126E-03 0.00860  3.20353E-04 0.01940  6.46482E-05 0.04210 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.71040E-01 0.01582  1.24754E-02 3.2E-05  3.22921E-02 0.00012  1.06336E-01 0.00055  2.98785E-01 0.00029  1.24219E+00 0.00339  5.55726E+00 0.03398 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.30528E-03 0.00784  2.63407E-04 0.02709  7.87334E-04 0.01657  6.08325E-04 0.01798  1.29205E-03 0.01307  2.95113E-04 0.02804  5.90465E-05 0.06375 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.77608E-01 0.02376  1.24756E-02 4.0E-05  3.22960E-02 0.00015  1.06276E-01 0.00080  2.98712E-01 0.00038  1.24569E+00 0.00072  6.49181E+00 0.02634 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.83999E-07 0.00127  8.83717E-07 0.00127  9.67098E-07 0.01752 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  8.10026E-07 0.00123  8.09767E-07 0.00123  8.86071E-07 0.01750 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.32515E-03 0.00909  2.71207E-04 0.03385  8.07163E-04 0.01861  5.93500E-04 0.02091  1.30555E-03 0.01426  2.90724E-04 0.03141  5.70058E-05 0.07590 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.71144E-01 0.02644  1.24747E-02 6.4E-05  3.23020E-02 0.00022  1.06384E-01 0.00101  2.98609E-01 0.00047  1.24719E+00 0.00102  6.91325E+00 0.03586 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.86789E-07 0.01976  7.86461E-07 0.01976  8.81650E-07 0.06111 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  7.20984E-07 0.01975  7.20685E-07 0.01975  8.07690E-07 0.06114 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.04495E-03 0.04395  2.32109E-04 0.13612  6.77484E-04 0.07627  5.39087E-04 0.09535  1.19248E-03 0.06697  3.02050E-04 0.13245  1.01739E-04 0.26514 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.65358E-01 0.12778  1.24752E-02 0.00017  3.22877E-02 0.00066  1.06535E-01 0.00318  2.97977E-01 0.00164  1.24662E+00 0.00299  6.46001E+00 0.10089 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.03334E-03 0.04324  2.30117E-04 0.13091  6.78202E-04 0.07561  5.43106E-04 0.09365  1.17909E-03 0.06576  3.02540E-04 0.13311  1.00284E-04 0.26013 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.70142E-01 0.12787  1.24752E-02 0.00017  3.22882E-02 0.00064  1.06530E-01 0.00317  2.98046E-01 0.00166  1.24638E+00 0.00294  6.46001E+00 0.10089 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.89754E+03 0.03960 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.77538E-07 0.00072 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  8.04097E-07 0.00059 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.29100E-03 0.00596 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.75028E+03 0.00592 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.80109E-09 0.00075 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.70808E-05 0.00846  6.70735E-05 0.00851  1.12260E-06 0.52180 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.91950E-05 0.03210  5.93848E-05 0.03230  4.15460E-07 0.59476 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.49953E-04 0.02533  1.50108E-04 0.02543  1.06536E-04 0.49839 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.68879E+01 0.01107 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.51201E+01 0.00027  4.02786E+01 0.00036 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:16:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 12:16:08 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615384667 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01297E+00  9.98292E-01  9.93878E-01  1.00133E+00  9.92918E-01  9.98106E-01  1.00409E+00  9.98551E-01  9.99693E-01  1.00075E+00  9.96734E-01  9.98929E-01  1.00373E+00  9.96386E-01  9.96504E-01  9.96969E-01  1.00180E+00  1.00837E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.96546E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70345E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39922E-01 0.00017  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53658E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59645E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.49812E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48904E+01 0.00028  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.20416E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.24345E+00 0.00051  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334572 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33414E+04 0.00066 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33414E+04 0.00066 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  3.87836E+03 ;
RUNNING_TIME              (idx, 1)        =  4.19737E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.81050E-01  9.81050E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.21983E+00  1.39000E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.17252E+02  5.17958E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.55683E-01  1.88500E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.33388E+01  4.98413E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.14794E+02  7.23228E+02 ];
CPU_USAGE                 (idx, 1)        = 9.23998 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99112E+00 1.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.17269E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22056.57;
MEMSIZE                   (idx, 1)        = 21853.12;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 203.45;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.87876E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73160E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.49852E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.02022E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.39360E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.82477E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.55862E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  3.09956E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.77919E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.01740E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.20822E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.04692E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  3.85142E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.38508E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38424E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35290E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57384E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.87700E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.30880E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26707E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.91561E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  6.41791E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.10937E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.77881E+15 0.00037  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 8 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57631E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30900E+03  3.65900E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.20604E-02 -4.87087E+27  2.25668E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02757E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.58282E+18 0.00398  1.68798E-02 0.00390 ];
U233_FISS                 (idx, [1:   4]) = [  8.48030E+19 0.00054  9.04424E-01 0.00017 ];
U235_FISS                 (idx, [1:   4]) = [  4.64681E+18 0.00242  4.95581E-02 0.00237 ];
PU239_FISS                (idx, [1:   4]) = [  1.99833E+16 0.03663  2.13136E-04 0.03666 ];
PU240_FISS                (idx, [1:   4]) = [  4.90183E+14 0.22241  5.24756E-06 0.22242 ];
PU241_FISS                (idx, [1:   4]) = [  3.60798E+14 0.27985  3.84909E-06 0.27990 ];
TH232_CAPT                (idx, [1:   4]) = [  9.88094E+19 0.00058  6.94657E-01 0.00030 ];
U233_CAPT                 (idx, [1:   4]) = [  1.23868E+19 0.00137  8.70850E-02 0.00135 ];
U235_CAPT                 (idx, [1:   4]) = [  1.63877E+18 0.00401  1.15212E-02 0.00400 ];
U238_CAPT                 (idx, [1:   4]) = [  5.13381E+13 0.70594  3.62264E-07 0.70599 ];
PU239_CAPT                (idx, [1:   4]) = [  1.07264E+16 0.04966  7.53920E-05 0.04965 ];
PU240_CAPT                (idx, [1:   4]) = [  1.81729E+15 0.11439  1.27576E-05 0.11438 ];
PU241_CAPT                (idx, [1:   4]) = [  1.29700E+14 0.44422  9.10567E-07 0.44422 ];
SM149_CAPT                (idx, [1:   4]) = [  1.71779E+17 0.01258  1.20760E-03 0.01255 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002410 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.10128E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002410 1.00110E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5480886 5.48575E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3613178 3.61619E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 908346 9.09071E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002410 1.00110E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 9.68575E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36159E+20 3.9E-06  2.36159E+20 3.9E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.37033E+19 1.1E-07  9.37033E+19 1.1E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42184E+20 0.00029  1.27578E+20 0.00019  1.46055E+19 0.00187 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.35887E+20 0.00018  2.21282E+20 0.00011  1.46055E+19 0.00187 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59294E+20 0.00037  2.59294E+20 0.00037  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.44100E+22 0.00026  3.26467E+22 0.00023  1.76325E+21 0.00180 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.35723E+19 0.00151 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59459E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16520E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.42392E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.42392E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.95840E-01 0.05494 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.24736E-01 0.01404 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.54388E-04 0.02613 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.99285E+03 0.02693 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09095E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.6E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.51514E-01 0.04244 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  5.92346E-01 0.04244 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52029E+00 3.8E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99828E+02 1.1E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.11466E-01 0.00045  1.00927E-01 0.00043  3.40795E-04 0.00916 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.11225E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.10816E-01 0.00037 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.11225E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00239E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.92432E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.92379E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.47578E-02 0.00139 ];
IMP_EALF                  (idx, [1:   2]) = [  1.47633E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.83906E-01 0.00141 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.84061E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.75685E-03 0.00550  2.88560E-04 0.01835  8.97454E-04 0.01011  6.79727E-04 0.01333  1.49326E-03 0.00913  3.26623E-04 0.01823  7.12235E-05 0.04059 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.84276E-01 0.01536  1.24759E-02 2.9E-05  3.22669E-02 0.00012  1.06509E-01 0.00052  3.00041E-01 0.00030  1.25349E+00 0.00066  5.89941E+00 0.03127 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.35510E-03 0.00768  2.57988E-04 0.02831  7.90575E-04 0.01617  6.00775E-04 0.01854  1.35222E-03 0.01277  2.90914E-04 0.02694  6.26311E-05 0.06152 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.81160E-01 0.02098  1.24761E-02 3.6E-05  3.22673E-02 0.00016  1.06417E-01 0.00075  3.00007E-01 0.00041  1.25356E+00 0.00087  6.69229E+00 0.02431 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.64119E-07 0.00132  8.63813E-07 0.00132  9.56446E-07 0.01595 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.87571E-07 0.00125  7.87292E-07 0.00125  8.71737E-07 0.01595 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.36642E-03 0.00925  2.62297E-04 0.03056  7.98804E-04 0.01760  6.03119E-04 0.02317  1.33298E-03 0.01383  3.05423E-04 0.02797  6.37962E-05 0.06473 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.89267E-01 0.02549  1.24756E-02 5.8E-05  3.22647E-02 0.00022  1.06397E-01 0.00094  3.00302E-01 0.00050  1.25205E+00 0.00121  6.69227E+00 0.03481 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.63690E-07 0.01972  7.63240E-07 0.01972  9.41923E-07 0.05758 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.96241E-07 0.01972  6.95830E-07 0.01972  8.59219E-07 0.05768 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.95625E-03 0.04497  2.00255E-04 0.14625  7.46505E-04 0.07969  5.39973E-04 0.10305  1.12541E-03 0.06324  2.70464E-04 0.12222  7.36497E-05 0.23325 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.41411E-01 0.09296  1.24775E-02 0.00013  3.22553E-02 0.00082  1.06434E-01 0.00313  3.00292E-01 0.00199  1.25239E+00 0.00327  7.06592E+00 0.08440 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.96588E-03 0.04413  1.89603E-04 0.14535  7.43165E-04 0.07768  5.49242E-04 0.10108  1.13399E-03 0.06211  2.77900E-04 0.12195  7.19826E-05 0.23850 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.29591E-01 0.08730  1.24775E-02 0.00013  3.22554E-02 0.00081  1.06429E-01 0.00312  3.00301E-01 0.00198  1.25229E+00 0.00327  7.08331E+00 0.08431 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -3.89319E+03 0.04088 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.56818E-07 0.00075 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.80914E-07 0.00060 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.38835E-03 0.00578 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -3.95600E+03 0.00593 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.76898E-09 0.00076 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.77855E-05 0.00870  6.77491E-05 0.00864  8.98431E-07 0.62943 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.67610E-05 0.02812  5.66488E-05 0.02812  1.33115E-06 0.57619 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.52536E-04 0.02518  1.52812E-04 0.02523  8.50833E-05 0.57563 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.67352E+01 0.01074 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48904E+01 0.00028  3.97222E+01 0.00036 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:16:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 13:08:02 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615384667 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01880E+00  9.98237E-01  9.96772E-01  9.99893E-01  1.00297E+00  9.98796E-01  1.00029E+00  9.99374E-01  9.96420E-01  9.94935E-01  9.97811E-01  9.94029E-01  9.99922E-01  1.00033E+00  9.99472E-01  9.98933E-01  1.00077E+00  1.00223E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.97050E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70295E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39728E-01 0.00016  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53487E-01 0.00015  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59891E+00 0.00029  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48914E+01 0.00024  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.48006E+01 0.00024  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.19388E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.24455E+00 0.00053  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334094 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33410E+04 0.00071 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33410E+04 0.00071 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.35153E+03 ;
RUNNING_TIME              (idx, 1)        =  4.71628E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.81050E-01  9.81050E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.36095E+00  1.41117E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  4.68955E+02  5.17026E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.75317E-01  1.96333E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  0.00000E+00  0.00000E+00 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  3.82687E+01  4.92987E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  4.66739E+02  7.26521E+02 ];
CPU_USAGE                 (idx, 1)        = 9.22660 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99116E+00 1.5E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.15773E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22056.57;
MEMSIZE                   (idx, 1)        = 21853.12;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 203.45;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 182 ;

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

TOT_ACTIVITY              (idx, 1)        =  6.90066E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73763E+08 ;
TOT_SF_RATE               (idx, 1)        =  3.45866E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  2.01221E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.38738E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.85440E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56500E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  4.02119E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  4.98866E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  2.84569E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.37743E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.13531E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.04300E+11 ;
SR90_ACTIVITY             (idx, 1)        =  2.82120E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.38480E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.35197E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.57328E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.92730E+14 ;
CS137_ACTIVITY            (idx, 1)        =  2.73854E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.27480E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.96728E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  7.83465E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.11692E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.77298E+15 0.00032  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 9 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.21959E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  9.13400E+03  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -2.71692E-02 -5.99888E+27  2.26796E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02609E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.58421E+18 0.00442  1.69122E-02 0.00441 ];
U233_FISS                 (idx, [1:   4]) = [  8.36301E+19 0.00058  8.92764E-01 0.00019 ];
U235_FISS                 (idx, [1:   4]) = [  5.36696E+18 0.00243  5.72909E-02 0.00231 ];
PU239_FISS                (idx, [1:   4]) = [  4.40061E+16 0.02503  4.69767E-04 0.02505 ];
PU240_FISS                (idx, [1:   4]) = [  1.50095E+15 0.13043  1.60202E-05 0.13038 ];
PU241_FISS                (idx, [1:   4]) = [  1.76349E+15 0.11646  1.88572E-05 0.11641 ];
TH232_CAPT                (idx, [1:   4]) = [  9.76101E+19 0.00050  6.86659E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.22023E+19 0.00145  8.58397E-02 0.00139 ];
U235_CAPT                 (idx, [1:   4]) = [  1.89382E+18 0.00362  1.33229E-02 0.00363 ];
U238_CAPT                 (idx, [1:   4]) = [  2.83607E+14 0.29645  1.99621E-06 0.29644 ];
PU239_CAPT                (idx, [1:   4]) = [  2.24702E+16 0.03277  1.58102E-04 0.03280 ];
PU240_CAPT                (idx, [1:   4]) = [  3.98611E+15 0.08265  2.80654E-05 0.08267 ];
PU241_CAPT                (idx, [1:   4]) = [  3.38544E+14 0.27175  2.38065E-06 0.27175 ];
SM149_CAPT                (idx, [1:   4]) = [  1.77627E+17 0.01208  1.24963E-03 0.01208 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002293 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.12211E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002293 1.00112E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5481378 5.48644E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3612257 3.61543E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 908658 9.09352E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002293 1.00112E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.32831E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36146E+20 4.2E-06  2.36146E+20 4.2E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36891E+19 1.0E-07  9.36891E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42175E+20 0.00028  1.27630E+20 0.00020  1.45447E+19 0.00180 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.35864E+20 0.00017  2.21319E+20 0.00011  1.45447E+19 0.00180 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59099E+20 0.00032  2.59099E+20 0.00032  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.43544E+22 0.00023  3.26002E+22 0.00021  1.75410E+21 0.00166 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.35616E+19 0.00137 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59426E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16202E+22 0.00030 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.39574E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.39574E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.91419E-01 0.05207 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.22352E-01 0.01479 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.53945E-04 0.02618 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.33163E+03 0.02635 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09067E-01 0.00013 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99998E-01 4.5E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.85136E-01 0.03937 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.22923E-01 0.03937 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52052E+00 4.1E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99858E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.11191E-01 0.00047  1.00905E-01 0.00045  3.46144E-04 0.00865 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.11290E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.11437E-01 0.00032 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.11290E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00243E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.90277E+00 0.00019 ];
IMP_ALF                   (idx, [1:   2]) = [  6.90360E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.50790E-02 0.00133 ];
IMP_EALF                  (idx, [1:   2]) = [  1.50644E-02 0.00094 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.88814E-01 0.00158 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.88721E-01 0.00046 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.82351E-03 0.00583  2.84289E-04 0.02167  8.82963E-04 0.01183  6.96386E-04 0.01280  1.53512E-03 0.00908  3.53838E-04 0.01644  7.09216E-05 0.04318 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.98583E-01 0.01669  1.24340E-02 0.00334  3.22543E-02 0.00014  1.06520E-01 0.00056  3.00371E-01 0.00032  1.25489E+00 0.00069  6.04510E+00 0.03192 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.43954E-03 0.00887  2.54049E-04 0.03183  7.92298E-04 0.01803  6.32337E-04 0.01927  1.39254E-03 0.01281  3.04358E-04 0.02595  6.39567E-05 0.06109 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  3.95336E-01 0.02245  1.24752E-02 4.9E-05  3.22580E-02 0.00020  1.06446E-01 0.00077  3.00418E-01 0.00043  1.25557E+00 0.00098  7.13181E+00 0.02264 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.57222E-07 0.00128  8.56775E-07 0.00127  9.89203E-07 0.01642 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.81036E-07 0.00117  7.80628E-07 0.00116  9.01527E-07 0.01648 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.41247E-03 0.00882  2.50572E-04 0.03543  7.96819E-04 0.01894  6.37216E-04 0.02071  1.35780E-03 0.01446  3.10932E-04 0.02971  5.91328E-05 0.06894 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.86317E-01 0.02464  1.24758E-02 6.2E-05  3.22487E-02 0.00023  1.06363E-01 0.00099  3.00348E-01 0.00054  1.25506E+00 0.00130  7.17401E+00 0.03298 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.58392E-07 0.01961  7.58171E-07 0.01961  8.25375E-07 0.05367 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.91091E-07 0.01961  6.90889E-07 0.01961  7.51932E-07 0.05359 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.16904E-03 0.04200  2.10643E-04 0.14675  7.66017E-04 0.07395  6.19761E-04 0.09199  1.24412E-03 0.06743  2.66588E-04 0.12709  6.19123E-05 0.40320 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.66942E-01 0.11212  1.24739E-02 0.00019  3.22441E-02 0.00076  1.06891E-01 0.00367  3.00147E-01 0.00196  1.25378E+00 0.00364  7.16777E+00 0.09957 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.18465E-03 0.04181  2.00390E-04 0.14067  7.75720E-04 0.07438  6.38633E-04 0.09011  1.24763E-03 0.06700  2.59782E-04 0.12595  6.24909E-05 0.39677 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.62292E-01 0.10897  1.24740E-02 0.00019  3.22445E-02 0.00076  1.06938E-01 0.00369  3.00163E-01 0.00193  1.25381E+00 0.00363  7.16777E+00 0.09957 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.17607E+03 0.03692 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.50627E-07 0.00075 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.75034E-07 0.00060 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.46102E-03 0.00546 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.06907E+03 0.00545 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.75966E-09 0.00081 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.68087E-05 0.00898  6.67787E-05 0.00899  1.83532E-06 0.36324 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.76398E-05 0.02709  5.77019E-05 0.02706  1.18285E-06 0.51111 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.51234E-04 0.02492  1.51006E-04 0.02500  2.16768E-04 0.35011 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.62857E+01 0.01118 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.48006E+01 0.00024  3.95204E+01 0.00035 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:16:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 14:01:01 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615384667 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01705E+00  9.97101E-01  9.97650E-01  1.00255E+00  1.00383E+00  9.95254E-01  1.00069E+00  9.94622E-01  9.98664E-01  9.98846E-01  9.96994E-01  9.99693E-01  9.99576E-01  9.94853E-01  1.00275E+00  9.96131E-01  1.00489E+00  9.98860E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.97264E-02 0.00031  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70274E-01 9.4E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.40003E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53763E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.59664E+00 0.00030  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.48379E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.47472E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.17425E+01 0.00030  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.24367E+00 0.00048  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333514 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33408E+04 0.00058 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33408E+04 0.00058 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  4.83485E+03 ;
RUNNING_TIME              (idx, 1)        =  5.24621E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.81050E-01  9.81050E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.50327E+00  1.42317E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.21755E+02  5.27999E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  1.93933E-01  1.86167E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.36333E-02  1.36333E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.32890E+01  5.02027E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.19641E+02  7.27095E+02 ];
CPU_USAGE                 (idx, 1)        = 9.21590 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99131E+00 1.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.14573E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22056.57;
MEMSIZE                   (idx, 1)        = 21853.12;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 203.45;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.90213E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73840E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.06231E+07 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.99868E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.37787E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.86943E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.56677E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  5.08202E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.16986E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  3.82652E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  9.54831E+10 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.21219E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.20648E+11 ;
SR90_ACTIVITY             (idx, 1)        =  3.20580E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37678E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.34465E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.56394E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.95060E+14 ;
CS137_ACTIVITY            (idx, 1)        =  3.12036E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26822E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  5.99151E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  9.16062E+16 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.09924E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.77445E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 10 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  3.86287E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.09590E+04  1.82500E+03 ];
FIMA                      (idx, [1:  3])  = [ -3.23304E-02 -7.13846E+27  2.27936E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02357E+00 0.00070 ];
TH232_FISS                (idx, [1:   4]) = [  1.56453E+18 0.00434  1.66983E-02 0.00425 ];
U233_FISS                 (idx, [1:   4]) = [  8.28111E+19 0.00052  8.83908E-01 0.00020 ];
U235_FISS                 (idx, [1:   4]) = [  5.85855E+18 0.00202  6.25330E-02 0.00196 ];
PU239_FISS                (idx, [1:   4]) = [  8.24986E+16 0.01711  8.80681E-04 0.01713 ];
PU240_FISS                (idx, [1:   4]) = [  3.99293E+15 0.07836  4.25993E-05 0.07829 ];
PU241_FISS                (idx, [1:   4]) = [  4.40085E+15 0.07442  4.69623E-05 0.07443 ];
TH232_CAPT                (idx, [1:   4]) = [  9.65516E+19 0.00052  6.78952E-01 0.00033 ];
U233_CAPT                 (idx, [1:   4]) = [  1.20928E+19 0.00156  8.50362E-02 0.00150 ];
U235_CAPT                 (idx, [1:   4]) = [  2.05856E+18 0.00345  1.44763E-02 0.00346 ];
U238_CAPT                 (idx, [1:   4]) = [  3.87975E+14 0.25210  2.73630E-06 0.25209 ];
PU239_CAPT                (idx, [1:   4]) = [  3.79248E+16 0.02518  2.66655E-04 0.02518 ];
PU240_CAPT                (idx, [1:   4]) = [  1.04723E+16 0.05293  7.36519E-05 0.05298 ];
PU241_CAPT                (idx, [1:   4]) = [  9.32195E+14 0.17083  6.54934E-06 0.17060 ];
SM149_CAPT                (idx, [1:   4]) = [  1.73041E+17 0.01271  1.21701E-03 0.01274 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002249 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.09895E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002249 1.00110E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5482445 5.48751E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3612296 3.61524E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 907508 9.08232E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002249 1.00110E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 1.91852E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36154E+20 4.6E-06  2.36154E+20 4.6E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36766E+19 1.0E-07  9.36766E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42195E+20 0.00029  1.27558E+20 0.00019  1.46372E+19 0.00192 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.35872E+20 0.00017  2.21234E+20 0.00011  1.46372E+19 0.00192 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59148E+20 0.00034  2.59148E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.43248E+22 0.00024  3.25562E+22 0.00021  1.76853E+21 0.00191 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.35375E+19 0.00149 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.59409E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16083E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.36792E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.36792E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.81836E-01 0.05581 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.36128E-01 0.01358 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.54351E-04 0.02550 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.33694E+03 0.02661 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.09179E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.2E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.61252E-01 0.04151 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.01169E-01 0.04151 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52095E+00 4.6E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99885E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.11314E-01 0.00044  1.00920E-01 0.00042  3.44352E-04 0.00956 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.11370E-01 0.00020 ];
COL_KEFF                  (idx, [1:   2]) = [  9.11301E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.11370E-01 0.00020 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00242E+00 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.88437E+00 0.00020 ];
IMP_ALF                   (idx, [1:   2]) = [  6.88594E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.53594E-02 0.00141 ];
IMP_EALF                  (idx, [1:   2]) = [  1.53327E-02 0.00089 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  2.92469E-01 0.00150 ];
IMP_AFGE                  (idx, [1:   2]) = [  2.92519E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.85691E-03 0.00580  2.79780E-04 0.02049  8.98660E-04 0.01126  6.82644E-04 0.01342  1.57454E-03 0.00859  3.47702E-04 0.01832  7.35848E-05 0.03759 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  3.97315E-01 0.01473  1.24768E-02 4.1E-05  3.22495E-02 0.00013  1.06656E-01 0.00051  3.00604E-01 0.00029  1.25871E+00 0.00078  6.18951E+00 0.02842 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.45588E-03 0.00788  2.46170E-04 0.02768  7.95484E-04 0.01562  6.05981E-04 0.02110  1.42728E-03 0.01178  3.10225E-04 0.02677  7.07400E-05 0.05745 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.02066E-01 0.02135  1.24766E-02 4.4E-05  3.22556E-02 0.00019  1.06532E-01 0.00070  3.00576E-01 0.00042  1.25926E+00 0.00107  6.82394E+00 0.02285 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.48882E-07 0.00131  8.48428E-07 0.00131  9.80046E-07 0.01716 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.73565E-07 0.00127  7.73151E-07 0.00127  8.93147E-07 0.01717 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.40608E-03 0.00953  2.45421E-04 0.03514  8.04340E-04 0.01866  5.99135E-04 0.02120  1.39332E-03 0.01506  3.00122E-04 0.02900  6.37381E-05 0.06192 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  3.91208E-01 0.02481  1.24768E-02 4.5E-05  3.22445E-02 0.00021  1.06565E-01 0.00099  3.00877E-01 0.00051  1.25959E+00 0.00132  6.58344E+00 0.03407 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.45421E-07 0.01969  7.45116E-07 0.01969  7.75179E-07 0.05445 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.79542E-07 0.01969  6.79264E-07 0.01969  7.06840E-07 0.05447 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.16590E-03 0.04386  2.48267E-04 0.13144  6.84204E-04 0.08021  5.80157E-04 0.09326  1.29297E-03 0.06562  2.99944E-04 0.11684  6.03511E-05 0.30465 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.80429E-01 0.09114  1.24748E-02 0.00017  3.22462E-02 0.00076  1.06470E-01 0.00295  3.00737E-01 0.00195  1.25693E+00 0.00365  5.92414E+00 0.12241 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.15209E-03 0.04340  2.52922E-04 0.12923  6.68482E-04 0.07813  5.77099E-04 0.09328  1.29708E-03 0.06545  2.96689E-04 0.11440  5.98174E-05 0.30638 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.78349E-01 0.08950  1.24748E-02 0.00017  3.22507E-02 0.00076  1.06482E-01 0.00297  3.00823E-01 0.00195  1.25754E+00 0.00368  6.00234E+00 0.12017 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.28008E+03 0.03974 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.43178E-07 0.00076 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.68354E-07 0.00061 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.40004E-03 0.00641 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.03446E+03 0.00662 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.74659E-09 0.00071 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.55622E-05 0.00915  6.55516E-05 0.00913  1.42191E-06 0.43656 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.73743E-05 0.03121  5.74031E-05 0.03121  8.06694E-07 0.44596 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.51694E-04 0.02465  1.51675E-04 0.02446  1.70867E-04 0.40594 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.61107E+01 0.01178 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.47472E+01 0.00027  3.93550E+01 0.00038 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:16:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 14:54:51 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615384667 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01661E+00  1.00190E+00  9.97053E-01  9.97382E-01  9.96353E-01  1.00286E+00  1.00082E+00  9.99253E-01  1.00437E+00  1.00017E+00  9.93923E-01  9.91272E-01  1.00095E+00  9.99968E-01  9.95525E-01  9.99645E-01  1.00213E+00  9.99812E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  2.99008E-02 0.00032  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.70099E-01 9.9E-06  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.39227E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.53066E-01 0.00016  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.60332E+00 0.00032  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46518E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45605E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.16500E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.25206E+00 0.00049  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3334149 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33385E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33385E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.32656E+03 ;
RUNNING_TIME              (idx, 1)        =  5.78445E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.81050E-01  9.81050E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.65375E+00  1.50483E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  5.75375E+02  5.36202E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.12733E-01  1.88000E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.36333E-02  1.36333E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  4.82957E+01  5.00665E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  5.73483E+02  7.32390E+02 ];
CPU_USAGE                 (idx, 1)        = 9.20842 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99138E+00 1.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.13737E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22056.57;
MEMSIZE                   (idx, 1)        = 21853.12;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 203.45;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
NEUTRON_EMIN              (idx, 1)        =  1.00000E-09 ;
NEUTRON_EMAX              (idx, 1)        =  1.50000E+01 ;

% Unresolved resonance probability table sampling:

URES_DILU_CUT             (idx, 1)        =  1.00000E-09 ;
URES_EMIN                 (idx, 1)        =  1.00000E-06 ;
URES_EMAX                 (idx, 1)        =  9.81270E-01 ;
URES_AVAIL                (idx, 1)        = 224 ;
URES_USED                 (idx, 1)        = 187 ;

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

TOT_ACTIVITY              (idx, 1)        =  6.92194E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.74427E+08 ;
TOT_SF_RATE               (idx, 1)        =  6.33449E+08 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.96247E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.35604E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.92549E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57487E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  9.59854E+12 ;
INGESTION_TOXICITY        (idx, 1)        =  5.74804E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  8.10736E+12 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.03529E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.44250E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  4.70315E+11 ;
SR90_ACTIVITY             (idx, 1)        =  4.34438E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.37104E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.33741E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.55570E+18 ;
CS134_ACTIVITY            (idx, 1)        =  4.99030E+14 ;
CS137_ACTIVITY            (idx, 1)        =  4.27270E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.26297E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.04826E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  1.40652E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.07232E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.79718E+15 0.00035  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 11 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  6.43601E+02  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  1.82590E+04  7.30000E+03 ];
FIMA                      (idx, [1:  3])  = [ -5.28781E-02 -1.16753E+28  2.32472E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.02141E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.53859E+18 0.00427  1.64260E-02 0.00423 ];
U233_FISS                 (idx, [1:   4]) = [  8.08033E+19 0.00058  8.62660E-01 0.00021 ];
U235_FISS                 (idx, [1:   4]) = [  6.84688E+18 0.00187  7.30984E-02 0.00181 ];
U238_FISS                 (idx, [1:   4]) = [  7.70888E+13 0.57548  8.17085E-07 0.57552 ];
PU239_FISS                (idx, [1:   4]) = [  2.92775E+17 0.00994  3.12579E-03 0.00995 ];
PU240_FISS                (idx, [1:   4]) = [  2.13568E+16 0.03174  2.28020E-04 0.03173 ];
PU241_FISS                (idx, [1:   4]) = [  3.04490E+16 0.03078  3.25039E-04 0.03083 ];
TH232_CAPT                (idx, [1:   4]) = [  9.44773E+19 0.00051  6.61754E-01 0.00031 ];
U233_CAPT                 (idx, [1:   4]) = [  1.17774E+19 0.00151  8.24952E-02 0.00150 ];
U235_CAPT                 (idx, [1:   4]) = [  2.41586E+18 0.00353  1.69220E-02 0.00353 ];
U238_CAPT                 (idx, [1:   4]) = [  9.86029E+14 0.16076  6.91800E-06 0.16076 ];
PU239_CAPT                (idx, [1:   4]) = [  1.34639E+17 0.01384  9.43255E-04 0.01389 ];
PU240_CAPT                (idx, [1:   4]) = [  5.68480E+16 0.02036  3.98221E-04 0.02038 ];
PU241_CAPT                (idx, [1:   4]) = [  6.83877E+15 0.06001  4.79239E-05 0.06003 ];
SM149_CAPT                (idx, [1:   4]) = [  1.72570E+17 0.01212  1.20885E-03 0.01213 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001562 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08801E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001562 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5487861 5.49311E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3600657 3.60392E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 913044 9.13851E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001562 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 -2.42144E-08 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36303E+20 4.5E-06  2.36303E+20 4.5E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.36375E+19 9.3E-08  9.36375E+19 9.3E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.42772E+20 0.00029  1.28144E+20 0.00020  1.46279E+19 0.00192 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.36409E+20 0.00018  2.21781E+20 0.00012  1.46279E+19 0.00192 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.59906E+20 0.00035  2.59906E+20 0.00035  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.44087E+22 0.00025  3.26457E+22 0.00021  1.76293E+21 0.00182 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.37523E+19 0.00154 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60162E+20 0.00021 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.15934E+22 0.00033 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.26428E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.26428E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.56492E-01 0.05476 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.27034E-01 0.01284 ];
SIX_FF_P                  (idx, [1:   2]) = [  1.64425E-04 0.02576 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  9.20902E+03 0.02583 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.08617E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 5.4E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.70307E-01 0.04059 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.09058E-01 0.04059 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52359E+00 4.4E-06 ];
FISSE                     (idx, [1:   2]) = [  1.99968E+02 9.3E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.09469E-01 0.00045  1.00710E-01 0.00045  3.44949E-04 0.00887 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.09299E-01 0.00021 ];
COL_KEFF                  (idx, [1:   2]) = [  9.09219E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.09299E-01 0.00021 ];
ABS_KINF                  (idx, [1:   2]) = [  1.00076E+00 0.00018 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.84220E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.84268E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.60213E-02 0.00144 ];
IMP_EALF                  (idx, [1:   2]) = [  1.60107E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.02914E-01 0.00138 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.02480E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.89086E-03 0.00564  2.85711E-04 0.01907  9.02816E-04 0.01146  7.05119E-04 0.01235  1.55670E-03 0.00847  3.58488E-04 0.01736  8.20332E-05 0.03818 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.13686E-01 0.01495  1.24792E-02 0.00011  3.22284E-02 0.00014  1.06639E-01 0.00055  3.01085E-01 0.00031  1.26159E+00 0.00078  6.52076E+00 0.02604 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.43992E-03 0.00814  2.46303E-04 0.02790  7.95504E-04 0.01669  6.24202E-04 0.01906  1.38381E-03 0.01229  3.17982E-04 0.02494  7.21217E-05 0.05542 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.16471E-01 0.02171  1.24801E-02 0.00017  3.22264E-02 0.00019  1.06635E-01 0.00074  3.00939E-01 0.00043  1.26184E+00 0.00111  7.16278E+00 0.02088 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.35199E-07 0.00133  8.34836E-07 0.00133  9.44446E-07 0.01754 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.59543E-07 0.00126  7.59213E-07 0.00126  8.58824E-07 0.01751 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.40781E-03 0.00884  2.46936E-04 0.03294  7.93295E-04 0.01974  6.10559E-04 0.02197  1.37800E-03 0.01381  3.07154E-04 0.02864  7.18618E-05 0.06452 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.13583E-01 0.02535  1.24787E-02 0.00018  3.22209E-02 0.00024  1.06725E-01 0.00106  3.00995E-01 0.00052  1.26358E+00 0.00160  7.19430E+00 0.02875 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.37541E-07 0.01970  7.37226E-07 0.01970  7.74309E-07 0.05188 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.70885E-07 0.01969  6.70599E-07 0.01970  7.04164E-07 0.05185 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.02831E-03 0.04628  2.46557E-04 0.14687  6.67911E-04 0.08726  5.49656E-04 0.09674  1.25915E-03 0.06409  2.66459E-04 0.12751  3.85770E-05 0.29550 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.80776E-01 0.09147  1.24835E-02 0.00091  3.21929E-02 0.00094  1.06502E-01 0.00295  2.99586E-01 0.00174  1.26051E+00 0.00399  6.90192E+00 0.09368 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.02794E-03 0.04625  2.43325E-04 0.14498  6.81410E-04 0.08654  5.31405E-04 0.09705  1.26220E-03 0.06351  2.68364E-04 0.12419  4.12351E-05 0.28703 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.85123E-01 0.09278  1.24836E-02 0.00091  3.21948E-02 0.00093  1.06536E-01 0.00297  2.99659E-01 0.00173  1.26024E+00 0.00401  6.90192E+00 0.09368 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.13260E+03 0.04230 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.29665E-07 0.00079 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.54504E-07 0.00062 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.39114E-03 0.00558 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.08830E+03 0.00566 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.75882E-09 0.00079 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.72366E-05 0.00980  6.72093E-05 0.00982  1.59777E-06 0.38873 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  5.73469E-05 0.02731  5.74233E-05 0.02730  9.97661E-07 0.47887 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  1.64511E-04 0.02464  1.64455E-04 0.02463  1.72346E-04 0.37472 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.63273E+01 0.01117 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45605E+01 0.00027  3.89467E+01 0.00039 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:16:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 15:48:57 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615384667 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01680E+00  9.99426E-01  9.95682E-01  9.93551E-01  9.97603E-01  9.95398E-01  1.00296E+00  9.95442E-01  1.00217E+00  1.00266E+00  9.96466E-01  1.00055E+00  9.96961E-01  9.97819E-01  1.00151E+00  1.00580E+00  1.00558E+00  9.93615E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.03019E-02 0.00033  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.69698E-01 1.0E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.37800E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.51802E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.61794E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.46305E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.45385E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.20629E+01 0.00029  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.28282E+00 0.00050  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333870 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33398E+04 0.00061 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33398E+04 0.00061 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  5.82051E+03 ;
RUNNING_TIME              (idx, 1)        =  6.32552E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.81050E-01  9.81050E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.80078E+00  1.47033E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.29279E+02  5.39046E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.31517E-01  1.87833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.36333E-02  1.36333E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.33638E+01  5.06800E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.27524E+02  7.35096E+02 ];
CPU_USAGE                 (idx, 1)        = 9.20164 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99142E+00 1.2E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.12981E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22056.57;
MEMSIZE                   (idx, 1)        = 21853.12;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 203.45;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.91180E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73936E+08 ;
TOT_SF_RATE               (idx, 1)        =  7.15223E+09 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.92247E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33631E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.95527E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57187E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.53604E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.38996E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.36132E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.14155E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.69239E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.23760E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.58109E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.34496E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31637E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.52759E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.02572E+14 ;
CS137_ACTIVITY            (idx, 1)        =  5.59780E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.23290E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.02942E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.14955E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  9.01482E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.81658E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 12 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.28688E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  3.65090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.06404E-01 -2.34938E+28  2.44291E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01843E+00 0.00071 ];
TH232_FISS                (idx, [1:   4]) = [  1.52173E+18 0.00403  1.62654E-02 0.00395 ];
U233_FISS                 (idx, [1:   4]) = [  7.93472E+19 0.00056  8.48168E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  7.10836E+18 0.00209  7.59828E-02 0.00201 ];
U238_FISS                 (idx, [1:   4]) = [  1.03652E+14 0.49750  1.11249E-06 0.49752 ];
PU239_FISS                (idx, [1:   4]) = [  6.09452E+17 0.00702  6.51526E-03 0.00706 ];
PU240_FISS                (idx, [1:   4]) = [  7.36581E+16 0.01893  7.87449E-04 0.01895 ];
PU241_FISS                (idx, [1:   4]) = [  1.23610E+17 0.01420  1.32105E-03 0.01414 ];
TH232_CAPT                (idx, [1:   4]) = [  9.26385E+19 0.00055  6.46442E-01 0.00035 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15396E+19 0.00152  8.05251E-02 0.00146 ];
U235_CAPT                 (idx, [1:   4]) = [  2.47800E+18 0.00319  1.72925E-02 0.00320 ];
U238_CAPT                 (idx, [1:   4]) = [  2.28882E+15 0.10193  1.59808E-05 0.10194 ];
PU239_CAPT                (idx, [1:   4]) = [  2.81498E+17 0.01040  1.96431E-03 0.01038 ];
PU240_CAPT                (idx, [1:   4]) = [  1.80973E+17 0.01205  1.26295E-03 0.01207 ];
PU241_CAPT                (idx, [1:   4]) = [  2.53405E+16 0.03411  1.76788E-04 0.03407 ];
SM149_CAPT                (idx, [1:   4]) = [  1.74267E+17 0.01175  1.21620E-03 0.01178 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001952 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08860E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001952 1.00109E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5494780 5.50007E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3587564 3.59052E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 919608 9.20296E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001952 1.00109E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 3.91155E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36605E+20 4.7E-06  2.36605E+20 4.7E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35966E+19 9.8E-08  9.35966E+19 9.8E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43420E+20 0.00028  1.28408E+20 0.00019  1.50125E+19 0.00188 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37017E+20 0.00017  2.22004E+20 0.00011  1.50125E+19 0.00188 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.60553E+20 0.00034  2.60553E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.47003E+22 0.00025  3.29069E+22 0.00022  1.79332E+21 0.00191 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.39789E+19 0.00146 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.60996E+20 0.00020 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16167E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  8.05293E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  8.05293E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.66801E-01 0.04989 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.33329E-01 0.01140 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.12584E-04 0.02234 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  8.53292E+03 0.02910 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.07973E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99997E-01 6.0E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  6.99972E-01 0.03786 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  6.35513E-01 0.03786 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52792E+00 4.6E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00056E+02 9.8E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.07529E-01 0.00043  1.00506E-01 0.00044  3.49859E-04 0.00862 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.07537E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.08121E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.07537E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.99459E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.81605E+00 0.00022 ];
IMP_ALF                   (idx, [1:   2]) = [  6.81611E+00 0.00013 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.64460E-02 0.00149 ];
IMP_EALF                  (idx, [1:   2]) = [  1.64416E-02 0.00088 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.09624E-01 0.00150 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.09068E-01 0.00045 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.94014E-03 0.00517  2.77079E-04 0.02018  9.14706E-04 0.01042  6.97776E-04 0.01262  1.59842E-03 0.00811  3.66985E-04 0.01725  8.51677E-05 0.03425 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.19583E-01 0.01355  1.24778E-02 7.2E-05  3.22027E-02 0.00016  1.06767E-01 0.00056  3.01335E-01 0.00030  1.25998E+00 0.00112  6.74193E+00 0.02214 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.48430E-03 0.00764  2.44238E-04 0.02920  8.03450E-04 0.01530  6.09672E-04 0.01872  1.41251E-03 0.01234  3.35203E-04 0.02549  7.92273E-05 0.05071 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.30642E-01 0.02049  1.24778E-02 8.3E-05  3.21985E-02 0.00022  1.06923E-01 0.00083  3.01311E-01 0.00044  1.26119E+00 0.00135  7.23051E+00 0.01869 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.31392E-07 0.00136  8.31022E-07 0.00136  9.36940E-07 0.01807 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.54467E-07 0.00128  7.54131E-07 0.00128  8.50386E-07 0.01813 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.46598E-03 0.00869  2.45668E-04 0.03294  8.06127E-04 0.01853  5.97716E-04 0.02013  1.40525E-03 0.01422  3.29176E-04 0.02814  8.20397E-05 0.05988 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.35361E-01 0.02420  1.24799E-02 0.00013  3.22022E-02 0.00030  1.06670E-01 0.00107  3.01290E-01 0.00054  1.26389E+00 0.00187  7.10092E+00 0.02774 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.33430E-07 0.01960  7.33026E-07 0.01961  8.30089E-07 0.05763 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.65702E-07 0.01960  6.65335E-07 0.01960  7.53511E-07 0.05763 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  3.05979E-03 0.04173  1.52954E-04 0.15735  7.78189E-04 0.07614  6.14400E-04 0.08750  1.16663E-03 0.06405  2.66733E-04 0.13518  8.08889E-05 0.23244 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  4.07464E-01 0.10450  1.24922E-02 0.00110  3.22300E-02 0.00101  1.07235E-01 0.00348  3.01097E-01 0.00197  1.26781E+00 0.00431  6.26835E+00 0.09301 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  3.07350E-03 0.04047  1.60782E-04 0.15387  7.93230E-04 0.07484  6.07306E-04 0.08709  1.14947E-03 0.06144  2.78412E-04 0.13131  8.42938E-05 0.23246 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  4.05449E-01 0.10076  1.24922E-02 0.00110  3.22298E-02 0.00099  1.07251E-01 0.00349  3.01115E-01 0.00196  1.26821E+00 0.00431  6.26835E+00 0.09301 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.20243E+03 0.03750 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.23481E-07 0.00066 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.47294E-07 0.00052 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.44714E-03 0.00582 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.18590E+03 0.00576 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.86657E-09 0.00078 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.78882E-05 0.00648  6.79111E-05 0.00650  1.32455E-06 0.43262 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.08368E-05 0.02377  6.08967E-05 0.02379  1.11339E-06 0.46795 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.09604E-04 0.02108  2.09831E-04 0.02118  1.52372E-04 0.40743 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.59939E+01 0.01192 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.45385E+01 0.00026  3.86882E+01 0.00041 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:16:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 16:43:11 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615384667 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.01629E+00  9.98835E-01  9.99648E-01  1.00280E+00  9.98972E-01  1.00311E+00  9.96370E-01  9.97345E-01  1.00261E+00  9.91822E-01  9.90479E-01  1.00107E+00  1.00082E+00  1.00230E+00  9.99408E-01  9.92832E-01  1.00407E+00  1.00122E+00  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.06828E-02 0.00035  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.69317E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.36635E-01 0.00019  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.50792E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.62959E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47257E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46333E+01 0.00026  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.26030E+01 0.00031  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.31387E+00 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333807 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33402E+04 0.00063 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33402E+04 0.00063 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.31600E+03 ;
RUNNING_TIME              (idx, 1)        =  6.86782E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.81050E-01  9.81050E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  1.95232E+00  1.51533E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  6.83301E+02  5.40215E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.50200E-01  1.86833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  1.36333E-02  1.36333E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  5.83936E+01  5.02980E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  6.81792E+02  7.35867E+02 ];
CPU_USAGE                 (idx, 1)        = 9.19652 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.99150E+00 1.3E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.12414E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22056.57;
MEMSIZE                   (idx, 1)        = 21853.12;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 203.45;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.90899E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73748E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.36247E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.91182E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.33113E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96305E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57044E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.64256E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.56543E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.46031E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16069E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.76426E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.39328E+11 ;
SR90_ACTIVITY             (idx, 1)        =  5.93053E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33743E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.31073E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51978E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.03757E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.00713E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.22302E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01757E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.35109E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.99759E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.83882E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 13 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  1.93017E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  5.47590E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -1.61215E-01 -3.55959E+28  2.56393E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01545E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.50889E+18 0.00415  1.61173E-02 0.00407 ];
U233_FISS                 (idx, [1:   4]) = [  7.92780E+19 0.00056  8.46852E-01 0.00022 ];
U235_FISS                 (idx, [1:   4]) = [  7.08005E+18 0.00185  7.56305E-02 0.00181 ];
U238_FISS                 (idx, [1:   4]) = [  1.83667E+14 0.37419  1.96283E-06 0.37420 ];
PU239_FISS                (idx, [1:   4]) = [  6.61242E+17 0.00632  7.06366E-03 0.00632 ];
PU240_FISS                (idx, [1:   4]) = [  8.43767E+16 0.01641  9.01240E-04 0.01639 ];
PU241_FISS                (idx, [1:   4]) = [  1.45405E+17 0.01429  1.55284E-03 0.01421 ];
TH232_CAPT                (idx, [1:   4]) = [  9.22465E+19 0.00057  6.41517E-01 0.00035 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15261E+19 0.00147  8.01579E-02 0.00143 ];
U235_CAPT                 (idx, [1:   4]) = [  2.49101E+18 0.00331  1.73233E-02 0.00328 ];
U238_CAPT                 (idx, [1:   4]) = [  2.95498E+15 0.08980  2.05516E-05 0.08983 ];
PU239_CAPT                (idx, [1:   4]) = [  3.06222E+17 0.00879  2.12971E-03 0.00880 ];
PU240_CAPT                (idx, [1:   4]) = [  2.19725E+17 0.01073  1.52820E-03 0.01073 ];
PU241_CAPT                (idx, [1:   4]) = [  3.15862E+16 0.02863  2.19704E-04 0.02864 ];
XE135_CAPT                (idx, [1:   4]) = [  2.56531E+13 1.00000  1.79966E-07 1.00000 ];
SM149_CAPT                (idx, [1:   4]) = [  1.83282E+17 0.01232  1.27483E-03 0.01237 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10002059 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.08278E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10002059 1.00108E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5498227 5.50318E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3579686 3.58277E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 924146 9.24884E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10002059 1.00108E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.88710E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36674E+20 4.8E-06  2.36674E+20 4.8E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35899E+19 9.8E-08  9.35899E+19 9.8E-08  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43717E+20 0.00027  1.28133E+20 0.00020  1.55843E+19 0.00171 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37307E+20 0.00017  2.21723E+20 0.00011  1.55843E+19 0.00171 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61294E+20 0.00034  2.61294E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.50587E+22 0.00025  3.32158E+22 0.00022  1.84284E+21 0.00179 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.41672E+19 0.00145 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61474E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16744E+22 0.00034 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  7.87149E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.87149E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  6.15909E-01 0.04225 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.08865E-01 0.01114 ];
SIX_FF_P                  (idx, [1:   2]) = [  2.57504E-04 0.02059 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.84874E+03 0.03001 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.07518E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99993E-01 8.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.02173E-01 0.02862 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.28044E-01 0.02862 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52885E+00 4.8E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00070E+02 9.8E-08 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.05791E-01 0.00044  1.00327E-01 0.00044  3.43099E-04 0.00886 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.06137E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.05809E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.06137E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.98524E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.81111E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.81074E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.65269E-02 0.00140 ];
IMP_EALF                  (idx, [1:   2]) = [  1.65303E-02 0.00095 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.10555E-01 0.00140 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.10332E-01 0.00047 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.91208E-03 0.00572  2.85070E-04 0.01754  9.08249E-04 0.01122  7.07541E-04 0.01213  1.56548E-03 0.00846  3.65257E-04 0.01846  8.04855E-05 0.03688 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.12339E-01 0.01459  1.24793E-02 7.5E-05  3.21934E-02 0.00016  1.06782E-01 0.00056  3.01510E-01 0.00032  1.25725E+00 0.00112  6.61399E+00 0.02413 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.43487E-03 0.00782  2.47416E-04 0.02622  8.09082E-04 0.01557  6.09570E-04 0.01860  1.37289E-03 0.01196  3.23162E-04 0.02430  7.27536E-05 0.05595 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.15954E-01 0.02065  1.24796E-02 0.00010  3.21890E-02 0.00021  1.06752E-01 0.00075  3.01656E-01 0.00044  1.25619E+00 0.00171  7.17414E+00 0.01912 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.38034E-07 0.00144  8.37503E-07 0.00146  9.91998E-07 0.04372 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.59030E-07 0.00135  7.58549E-07 0.00136  8.98558E-07 0.04363 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.40297E-03 0.00882  2.46884E-04 0.03379  7.93342E-04 0.01909  6.10169E-04 0.02167  1.36648E-03 0.01335  3.17346E-04 0.03189  6.87431E-05 0.06188 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.06450E-01 0.02397  1.24797E-02 0.00015  3.21780E-02 0.00028  1.06989E-01 0.00108  3.01512E-01 0.00053  1.25694E+00 0.00178  7.10764E+00 0.02945 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.32904E-07 0.01969  7.32622E-07 0.01970  7.93350E-07 0.05589 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.64094E-07 0.01969  6.63840E-07 0.01969  7.18266E-07 0.05576 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.98965E-03 0.04617  2.02127E-04 0.14856  7.01675E-04 0.08712  5.79423E-04 0.09756  1.23013E-03 0.06557  2.37733E-04 0.15623  3.85649E-05 0.32772 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.54664E-01 0.10887  1.24861E-02 0.00091  3.21939E-02 0.00090  1.06909E-01 0.00331  3.02840E-01 0.00210  1.25641E+00 0.00556  7.24993E+00 0.08982 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.98254E-03 0.04569  1.95403E-04 0.14658  7.14873E-04 0.08578  5.76630E-04 0.09629  1.21637E-03 0.06470  2.39657E-04 0.15217  3.96024E-05 0.33155 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.48904E-01 0.10413  1.24860E-02 0.00091  3.21966E-02 0.00091  1.06868E-01 0.00326  3.02914E-01 0.00207  1.25654E+00 0.00548  7.24417E+00 0.08969 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.10025E+03 0.04187 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.27394E-07 0.00080 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.49397E-07 0.00063 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.44609E-03 0.00611 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.16665E+03 0.00625 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  6.99025E-09 0.00091 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.83709E-05 0.00594  6.83877E-05 0.00599  1.67318E-06 0.39048 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.07935E-05 0.02069  6.07547E-05 0.02069  2.02486E-06 0.54578 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  2.58588E-04 0.01936  2.58799E-04 0.01938  2.14571E-04 0.39416 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.60770E+01 0.01099 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46333E+01 0.00026  3.86404E+01 0.00039 ];


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
INPUT_FILE_NAME           (idx, [1: 36])  = 'msfr_mix1_benchmark_burn_temperature' ;
WORKING_DIRECTORY         (idx, [1: 35])  = '/gv/data/cfx/tiago/doutorado/malha4' ;
HOSTNAME                  (idx, [1: 26])  = 'demoninho.usuarios.cdtn.br' ;
CPU_TYPE                  (idx, [1: 41])  = 'Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz' ;
CPU_MHZ                   (idx, 1)        = 184549427.0 ;
START_DATE                (idx, [1: 24])  = 'Fri Apr 22 05:16:24 2022' ;
COMPLETE_DATE             (idx, [1: 24])  = 'Fri Apr 22 17:12:34 2022' ;

% Run parameters:

POP                       (idx, 1)        = 100000 ;
CYCLES                    (idx, 1)        = 100 ;
SKIP                      (idx, 1)        = 10 ;
BATCH_INTERVAL            (idx, 1)        = 1 ;
SRC_NORM_MODE             (idx, 1)        = 2 ;
SEED                      (idx, 1)        = 1650615384667 ;
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
OMP_HISTORY_PROFILE       (idx, [1:  18]) = [  1.03734E+00  1.00106E+00  9.96614E-01  9.93733E-01  9.99628E-01  9.97967E-01  9.92655E-01  9.94796E-01  1.00022E+00  9.99221E-01  1.00351E+00  1.00317E+00  9.96109E-01  9.93111E-01  9.98300E-01  1.00247E+00  9.94605E-01  9.95492E-01  ];
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
ST_FRAC                   (idx, [1:   4]) = [  3.11817E-02 0.00034  0.00000E+00 0.0E+00 ];
DT_FRAC                   (idx, [1:   4]) = [  9.68818E-01 1.1E-05  0.00000E+00 0.0E+00 ];
DT_EFF                    (idx, [1:   4]) = [  3.35047E-01 0.00018  0.00000E+00 0.0E+00 ];
IFC_COL_EFF               (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_EFF          (idx, [1:   4]) = [  1.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
REA_SAMPLING_FAIL         (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_COL_EFF               (idx, [1:   4]) = [  3.49418E-01 0.00017  0.00000E+00 0.0E+00 ];
AVG_TRACKING_LOOPS        (idx, [1:   8]) = [  4.64172E+00 0.00031  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
AVG_TRACKS                (idx, [1:   4]) = [  4.47619E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_REAL_COL              (idx, [1:   4]) = [  4.46687E+01 0.00027  0.00000E+00 0.0E+00 ];
AVG_VIRT_COL              (idx, [1:   4]) = [  8.31693E+01 0.00032  0.00000E+00 0.0E+00 ];
AVG_SURF_CROSS            (idx, [1:   4]) = [  1.35129E+00 0.00052  0.00000E+00 0.0E+00 ];
LOST_PARTICLES            (idx, 1)        = 0 ;

% Run statistics:

CYCLE_IDX                 (idx, 1)        = 100 ;
SIMULATED_HISTORIES       (idx, 1)        = 3333704 ;
MEAN_POP_SIZE             (idx, [1:  2])  = [  3.33369E+04 0.00065 ];
MEAN_POP_WGT              (idx, [1:  2])  = [  3.33369E+04 0.00065 ];
SIMULATION_COMPLETED      (idx, 1)        = 1 ;

% Running times:

TOT_CPU_TIME              (idx, 1)        =  6.58532E+03 ;
RUNNING_TIME              (idx, 1)        =  7.16170E+02 ;
INIT_TIME                 (idx, [1:  2])  = [  9.81050E-01  9.81050E-01 ];
PROCESS_TIME              (idx, [1:  2])  = [  2.10420E+00  1.51883E-01 ];
TRANSPORT_CYCLE_TIME      (idx, [1:  3])  = [  7.12479E+02  2.91777E+01  0.00000E+00 ];
BURNUP_CYCLE_TIME         (idx, [1:  2])  = [  2.68283E-01  1.80833E-02 ];
BATEMAN_SOLUTION_TIME     (idx, [1:  2])  = [  2.65333E-02  1.29000E-02 ];
MPI_OVERHEAD_TIME         (idx, [1:  2])  = [  6.09561E+01  2.56242E+00 ];
ESTIMATED_RUNNING_TIME    (idx, [1:  2])  = [  7.13647E+02  7.13647E+02 ];
CPU_USAGE                 (idx, 1)        = 9.19519 ;
TRANSPORT_CPU_USAGE       (idx, [1:   2]) = [  9.98566E+00 2.0E-05 ];
OMP_PARALLEL_FRAC         (idx, 1)        =  9.12318E-01 ;

% Memory usage:

AVAIL_MEM                 (idx, 1)        = 32055.89 ;
ALLOC_MEMSIZE             (idx, 1)        = 22056.57;
MEMSIZE                   (idx, 1)        = 21853.12;
XS_MEMSIZE                (idx, 1)        = 21004.80;
MAT_MEMSIZE               (idx, 1)        = 535.53;
RES_MEMSIZE               (idx, 1)        = 12.11;
IFC_MEMSIZE               (idx, 1)        = 0.00;
MISC_MEMSIZE              (idx, 1)        = 300.69;
UNKNOWN_MEMSIZE           (idx, 1)        = 0.00;
UNUSED_MEMSIZE            (idx, 1)        = 203.45;

% Geometry parameters:

TOT_CELLS                 (idx, 1)        = 3 ;
UNION_CELLS               (idx, 1)        = 0 ;

% Neutron energy grid:

NEUTRON_ERG_TOL           (idx, 1)        =  0.00000E+00 ;
NEUTRON_ERG_NE            (idx, 1)        = 1078181 ;
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

TOT_ACTIVITY              (idx, 1)        =  6.90630E+20 ;
TOT_DECAY_HEAT            (idx, 1)        =  1.73702E+08 ;
TOT_SF_RATE               (idx, 1)        =  1.59140E+10 ;
ACTINIDE_ACTIVITY         (idx, 1)        =  1.90577E+20 ;
ACTINIDE_DECAY_HEAT       (idx, 1)        =  1.32759E+07 ;
FISSION_PRODUCT_ACTIVITY  (idx, 1)        =  4.96636E+20 ;
FISSION_PRODUCT_DECAY_HEAT(idx, 1)        =  1.57027E+08 ;
INHALATION_TOXICITY       (idx, 1)        =  1.65504E+13 ;
INGESTION_TOXICITY        (idx, 1)        =  6.61295E+11 ;
ACTINIDE_INH_TOX          (idx, 1)        =  1.47055E+13 ;
ACTINIDE_ING_TOX          (idx, 1)        =  1.16146E+11 ;
FISSION_PRODUCT_INH_TOX   (idx, 1)        =  1.78538E+12 ;
FISSION_PRODUCT_ING_TOX   (idx, 1)        =  5.43981E+11 ;
SR90_ACTIVITY             (idx, 1)        =  6.03082E+18 ;
TE132_ACTIVITY            (idx, 1)        =  4.33592E+18 ;
I131_ACTIVITY             (idx, 1)        =  3.30974E+18 ;
I132_ACTIVITY             (idx, 1)        =  4.51829E+18 ;
CS134_ACTIVITY            (idx, 1)        =  5.04333E+14 ;
CS137_ACTIVITY            (idx, 1)        =  6.13375E+18 ;
PHOTON_DECAY_SOURCE       (idx, 1)        =  5.21833E+20 ;
NEUTRON_DECAY_SOURCE      (idx, 1)        =  6.01509E+17 ;
ALPHA_DECAY_SOURCE        (idx, 1)        =  2.38529E+17 ;
ELECTRON_DECAY_SOURCE     (idx, 1)        =  8.98674E+20 ;

% Normalization coefficient:

NORM_COEF                 (idx, [1:   4]) = [  7.84495E+15 0.00034  0.00000E+00 0.0E+00 ];

% Parameters for burnup calculation:

BURN_MATERIALS            (idx, 1)        = 8 ;
BURN_MODE                 (idx, 1)        = 2 ;
BURN_STEP                 (idx, 1)        = 14 ;
BURN_RANDOMIZE_DATA       (idx, [1:  3])  = [ 0 0 0 ];
BURNUP                    (idx, [1:  2])  = [  2.57345E+03  0.00000E+00 ];
BURN_DAYS                 (idx, [1:  2])  = [  7.30090E+04  1.82500E+04 ];
FIMA                      (idx, [1:  3])  = [ -2.16453E-01 -4.77921E+28  2.68589E+29 ];

% Analog reaction rate estimators:

CONVERSION_RATIO          (idx, [1:   2]) = [  1.01242E+00 0.00072 ];
TH232_FISS                (idx, [1:   4]) = [  1.49535E+18 0.00418  1.59864E-02 0.00416 ];
U233_FISS                 (idx, [1:   4]) = [  7.91950E+19 0.00061  8.46630E-01 0.00023 ];
U235_FISS                 (idx, [1:   4]) = [  7.09412E+18 0.00187  7.58405E-02 0.00181 ];
U238_FISS                 (idx, [1:   4]) = [  2.34142E+14 0.32889  2.50331E-06 0.32888 ];
PU239_FISS                (idx, [1:   4]) = [  6.53456E+17 0.00632  6.98599E-03 0.00631 ];
PU240_FISS                (idx, [1:   4]) = [  8.53789E+16 0.01776  9.12787E-04 0.01775 ];
PU241_FISS                (idx, [1:   4]) = [  1.52666E+17 0.01347  1.63212E-03 0.01348 ];
TH232_CAPT                (idx, [1:   4]) = [  9.19033E+19 0.00052  6.38961E-01 0.00035 ];
U233_CAPT                 (idx, [1:   4]) = [  1.15382E+19 0.00148  8.02199E-02 0.00144 ];
U235_CAPT                 (idx, [1:   4]) = [  2.48060E+18 0.00362  1.72464E-02 0.00360 ];
U238_CAPT                 (idx, [1:   4]) = [  2.77916E+15 0.08895  1.93181E-05 0.08895 ];
PU239_CAPT                (idx, [1:   4]) = [  3.05676E+17 0.00930  2.12524E-03 0.00929 ];
PU240_CAPT                (idx, [1:   4]) = [  2.23323E+17 0.01107  1.55267E-03 0.01108 ];
PU241_CAPT                (idx, [1:   4]) = [  3.07538E+16 0.03132  2.13771E-04 0.03127 ];
SM149_CAPT                (idx, [1:   4]) = [  1.81727E+17 0.01177  1.26354E-03 0.01176 ];

% Neutron balance (particles/weight):

BALA_SRC_NEUTRON_SRC     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_FISS    (idx, [1:  2])  = [ 10001071 1.00000E+07 ];
BALA_SRC_NEUTRON_NXN     (idx, [1:  2])  = [ 0 1.06715E+04 ];
BALA_SRC_NEUTRON_VR      (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_SRC_NEUTRON_TOT     (idx, [1:  2])  = [ 10001071 1.00107E+07 ];

BALA_LOSS_NEUTRON_CAPT    (idx, [1:  2])  = [ 5494902 5.50038E+06 ];
BALA_LOSS_NEUTRON_FISS    (idx, [1:  2])  = [ 3573863 3.57714E+06 ];
BALA_LOSS_NEUTRON_LEAK    (idx, [1:  2])  = [ 932306 9.33150E+05 ];
BALA_LOSS_NEUTRON_CUT     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_ERR     (idx, [1:  2])  = [ 0 0.00000E+00 ];
BALA_LOSS_NEUTRON_TOT     (idx, [1:  2])  = [ 10001071 1.00107E+07 ];

BALA_NEUTRON_DIFF         (idx, [1:  2])  = [ 0 2.57045E-07 ];

% Normalized total reaction rates (neutrons):

TOT_POWER                 (idx, [1:   6]) = [  3.00000E+09 0.0E+00  3.00000E+09 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_POWDENS               (idx, [1:   6]) = [  3.52484E-02 0.0E+00  3.52484E-02 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_GENRATE               (idx, [1:   6]) = [  2.36683E+20 5.2E-06  2.36683E+20 5.2E-06  0.00000E+00 0.0E+00 ];
TOT_FISSRATE              (idx, [1:   6]) = [  9.35892E+19 1.0E-07  9.35892E+19 1.0E-07  0.00000E+00 0.0E+00 ];
TOT_CAPTRATE              (idx, [1:   6]) = [  1.43749E+20 0.00028  1.27832E+20 0.00020  1.59170E+19 0.00168 ];
TOT_ABSRATE               (idx, [1:   6]) = [  2.37338E+20 0.00017  2.21421E+20 0.00012  1.59170E+19 0.00168 ];
TOT_SRCRATE               (idx, [1:   6]) = [  2.61498E+20 0.00034  2.61498E+20 0.00034  0.00000E+00 0.0E+00 ];
TOT_FLUX                  (idx, [1:   6]) = [  3.53422E+22 0.00024  3.34869E+22 0.00022  1.85526E+21 0.00179 ];
TOT_PHOTON_PRODRATE       (idx, [1:   4]) = [  0.00000E+00 0.0E+00  0.00000E+00 0.0E+00 ];
TOT_LEAKRATE              (idx, [1:   2]) = [  2.44026E+19 0.00146 ];
ALBEDO_LEAKRATE           (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_LOSSRATE              (idx, [1:   2]) = [  2.61741E+20 0.00019 ];
TOT_CUTRATE               (idx, [1:   2]) = [  0.00000E+00 0.0E+00 ];
TOT_RR                    (idx, [1:   2]) = [  1.16926E+22 0.00031 ];
INI_FMASS                 (idx, 1)        =  8.51102E+04 ;
TOT_FMASS                 (idx, 1)        =  7.71006E+04 ;
INI_BURN_FMASS            (idx, 1)        =  8.51102E+04 ;
TOT_BURN_FMASS            (idx, 1)        =  7.71006E+04 ;

% Six-factor formula:

SIX_FF_ETA                (idx, [1:   2]) = [  5.79219E-01 0.04113 ];
SIX_FF_F                  (idx, [1:   2]) = [  8.05930E-01 0.01018 ];
SIX_FF_P                  (idx, [1:   2]) = [  3.11622E-04 0.01892 ];
SIX_FF_EPSILON            (idx, [1:   2]) = [  7.19726E+03 0.03273 ];
SIX_FF_LF                 (idx, [1:   2]) = [  9.06691E-01 0.00014 ];
SIX_FF_LT                 (idx, [1:   2]) = [  9.99994E-01 7.7E-07 ];
SIX_FF_KINF               (idx, [1:   2]) = [  8.08007E-01 0.02801 ];
SIX_FF_KEFF               (idx, [1:   2]) = [  7.32632E-01 0.02801 ];

% Fission neutron and energy production:

NUBAR                     (idx, [1:   2]) = [  2.52896E+00 5.1E-06 ];
FISSE                     (idx, [1:   2]) = [  2.00071E+02 1.0E-07 ];

% Criticality eigenvalues:

ANA_KEFF                  (idx, [1:   6]) = [  9.04590E-01 0.00047  1.00165E-01 0.00045  3.50487E-04 0.00897 ];
IMP_KEFF                  (idx, [1:   2]) = [  9.05245E-01 0.00019 ];
COL_KEFF                  (idx, [1:   2]) = [  9.05136E-01 0.00034 ];
ABS_KEFF                  (idx, [1:   2]) = [  9.05245E-01 0.00019 ];
ABS_KINF                  (idx, [1:   2]) = [  9.98427E-01 0.00017 ];
GEOM_ALBEDO               (idx, [1:   6]) = [  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00  1.00000E+00 0.0E+00 ];

% ALF (Average lethargy of neutrons causing fission):
% Based on E0 = 1.500000E+01 MeV

ANA_ALF                   (idx, [1:   2]) = [  6.81377E+00 0.00021 ];
IMP_ALF                   (idx, [1:   2]) = [  6.81516E+00 0.00014 ];

% EALF (Energy corresponding to average lethargy of neutrons causing fission):

ANA_EALF                  (idx, [1:   2]) = [  1.64831E-02 0.00141 ];
IMP_EALF                  (idx, [1:   2]) = [  1.64574E-02 0.00092 ];

% AFGE (Average energy of neutrons causing fission):

ANA_AFGE                  (idx, [1:   2]) = [  3.10381E-01 0.00150 ];
IMP_AFGE                  (idx, [1:   2]) = [  3.10181E-01 0.00049 ];

% Forward-weighted delayed neutron parameters:

PRECURSOR_GROUPS          (idx, 1)        = 6 ;
FWD_ANA_BETA_ZERO         (idx, [1:  14]) = [  3.95041E-03 0.00564  2.86200E-04 0.02027  9.30123E-04 0.01087  6.96802E-04 0.01222  1.58913E-03 0.00855  3.64614E-04 0.01820  8.35360E-05 0.03862 ];
FWD_ANA_LAMBDA            (idx, [1:  14]) = [  4.12646E-01 0.01473  1.24814E-02 0.00013  3.21992E-02 0.00015  1.06745E-01 0.00057  3.01159E-01 0.00031  1.25675E+00 0.00111  6.47807E+00 0.02526 ];

% Beta-eff using Meulekamp's method:

ADJ_MEULEKAMP_BETA_EFF    (idx, [1:  14]) = [  3.51078E-03 0.00784  2.50386E-04 0.02758  8.25115E-04 0.01616  6.19861E-04 0.01757  1.41647E-03 0.01162  3.28191E-04 0.02645  7.07585E-05 0.05278 ];
ADJ_MEULEKAMP_LAMBDA      (idx, [1:  14]) = [  4.05736E-01 0.01962  1.24805E-02 0.00015  3.21960E-02 0.00021  1.06901E-01 0.00080  3.01171E-01 0.00043  1.25663E+00 0.00163  7.05468E+00 0.02027 ];

% Adjoint weighted time constants using Nauchi's method:

IFP_CHAIN_LENGTH          (idx, 1)        = 20 ;
ADJ_NAUCHI_GEN_TIME       (idx, [1:   6]) = [  8.50608E-07 0.00162  8.49991E-07 0.00161  1.02585E-06 0.04089 ];
ADJ_NAUCHI_LIFETIME       (idx, [1:   6]) = [  7.69401E-07 0.00155  7.68843E-07 0.00155  9.27732E-07 0.04084 ];
ADJ_NAUCHI_BETA_EFF       (idx, [1:  14]) = [  3.48630E-03 0.00892  2.47155E-04 0.03499  8.16450E-04 0.01948  6.23242E-04 0.02220  1.41363E-03 0.01394  3.17030E-04 0.02867  6.87973E-05 0.06060 ];
ADJ_NAUCHI_LAMBDA         (idx, [1:  14]) = [  4.07185E-01 0.02448  1.24811E-02 0.00020  3.22055E-02 0.00026  1.06801E-01 0.00100  3.01216E-01 0.00059  1.25575E+00 0.00218  7.11294E+00 0.02936 ];

% Adjoint weighted time constants using IFP:

ADJ_IFP_GEN_TIME          (idx, [1:   6]) = [  7.40328E-07 0.01976  7.39901E-07 0.01976  8.61420E-07 0.05438 ];
ADJ_IFP_LIFETIME          (idx, [1:   6]) = [  6.69709E-07 0.01975  6.69323E-07 0.01975  7.79252E-07 0.05435 ];
ADJ_IFP_IMP_BETA_EFF      (idx, [1:  14]) = [  2.98609E-03 0.04353  1.81951E-04 0.16334  7.70939E-04 0.07593  5.35907E-04 0.09866  1.16559E-03 0.06780  2.85019E-04 0.12785  4.66908E-05 0.33887 ];
ADJ_IFP_IMP_LAMBDA        (idx, [1:  14]) = [  3.40202E-01 0.07707  1.24780E-02 0.00015  3.22510E-02 0.00088  1.06478E-01 0.00304  3.01191E-01 0.00207  1.26452E+00 0.00512  6.93621E+00 0.11633 ];
ADJ_IFP_ANA_BETA_EFF      (idx, [1:  14]) = [  2.99767E-03 0.04305  1.83201E-04 0.16150  7.72183E-04 0.07542  5.25306E-04 0.09663  1.18312E-03 0.06596  2.86789E-04 0.12682  4.70748E-05 0.32646 ];
ADJ_IFP_ANA_LAMBDA        (idx, [1:  14]) = [  3.50541E-01 0.08753  1.24780E-02 0.00015  3.22486E-02 0.00088  1.06485E-01 0.00306  3.01203E-01 0.00205  1.26448E+00 0.00511  6.92014E+00 0.11611 ];
ADJ_IFP_ROSSI_ALPHA       (idx, [1:   2]) = [ -4.07260E+03 0.03977 ];

% Adjoint weighted time constants using perturbation technique:

ADJ_PERT_GEN_TIME         (idx, [1:   2]) = [  8.40584E-07 0.00081 ];
ADJ_PERT_LIFETIME         (idx, [1:   2]) = [  7.60330E-07 0.00063 ];
ADJ_PERT_BETA_EFF         (idx, [1:   2]) = [  3.38747E-03 0.00620 ];
ADJ_PERT_ROSSI_ALPHA      (idx, [1:   2]) = [ -4.03143E+03 0.00634 ];

% Inverse neutron speed :

ANA_INV_SPD               (idx, [1:   2]) = [  7.14383E-09 0.00102 ];

% Analog slowing-down and thermal neutron lifetime (total/prompt/delayed):

ANA_SLOW_TIME             (idx, [1:   6]) = [  6.88286E-05 0.00533  6.88063E-05 0.00536  4.56587E-06 0.23260 ];
ANA_THERM_TIME            (idx, [1:   6]) = [  6.38955E-05 0.01950  6.38326E-05 0.01947  4.58058E-06 0.31786 ];
ANA_THERM_FRAC            (idx, [1:   6]) = [  3.07624E-04 0.01853  3.06934E-04 0.01870  4.82847E-04 0.22321 ];
ANA_DELAYED_EMTIME        (idx, [1:   2]) = [  1.64705E+01 0.01080 ];
ANA_MEAN_NCOL             (idx, [1:   4]) = [  4.46687E+01 0.00027  3.86552E+01 0.00036 ];

