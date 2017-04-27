##--------------------------------------------------------------------------
##   Author: Thomas Nowotny
##
##   Institute: Center for Computational Neuroscience and Robotics
##              University of Sussex
##              Falmer, Brighton BN1 9QJ, UK
##
##   email to:  T.Nowotny@sussex.ac.uk
##
##   initial version: 2010-02-07
##
##--------------------------------------------------------------------------

EXECUTABLE              :=classol_sim
SOURCES                 :=$(EXECUTABLE).cc
OPTIMIZATIONFLAGS       :=-O3 -ffast-math
NVCC_OPTIMIZATIONFLAGS  :=-use_fast_math -O3
LINK_FLAGS              := -lpthread -lopencv_core -lopencv_highgui -lopencv_imgproc
ifndef CPU_ONLY
    LINK_FLAGS += -lopencv_gpu
endif

include	$(GENN_PATH)/userproject/include/makefile_common_gnu.mk
