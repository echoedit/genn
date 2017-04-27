/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Science
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file userproject/MBody1_project/model/classol_sim.cc

\brief Main entry point for the classol (CLASSification in OLfaction) model simulation. Provided as a part of the complete example of simulating the MBody1 mushroom body model. 
*/
//--------------------------------------------------------------------------



#include <chrono>
#include <thread>

#include <csignal>

// OpenCV includes
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifndef CPU_ONLY
#include <opencv2/gpu/gpu.hpp>
#endif  // CPU_ONLY

#include "classol_sim.h"

typedef std::chrono::high_resolution_clock sim_clock;

namespace
{
volatile std::sig_atomic_t g_SignalStatus;

void signalHandler(int status)
{
    g_SignalStatus = status;
}

//--------------------------------------------------------------------------
/*! \brief This function is the entry point for running the simulation of the MBody1 model network.
*/
//--------------------------------------------------------------------------

void gennThreadHandler(int which, FILE *osf, FILE *osf2, classol &locust)
{
    typedef std::chrono::duration<double, std::milli> double_ms;

    //------------------------------------------------------------------
    // output general parameters to output file and start the simulation

    fprintf(stdout, "# We are running with fixed time step %f \n", DT);
    t= 0.0;
    iT= 0;
    int done= 0;
    float last_t_report=  t;
    timer.startTimer();

    sim_clock::duration totalSleepTime{0};
    sim_clock::duration totalOverrunTime{0};

    const double_ms dtDurationMs{DT};
    const sim_clock::duration dtDuration = std::chrono::duration_cast<sim_clock::duration>(dtDurationMs);

#ifndef CPU_ONLY
    if (which == GPU){
        while (g_SignalStatus == 0)
        {
            // Get time step started at
            const auto stepStart = sim_clock::now();

            locust.runGPU(DT); // run next batch
            pullDNCurrentSpikesFromDevice();

#ifdef TIMING
            fprintf(timeros, "%f %f %f \n", neuron_tme, synapse_tme, learning_tme);
#endif
            locust.sum_spikes();
            locust.outputDN_spikes(osf2, which);

            fprintf(osf, "%f ", t);
            fprintf(osf,"\n");

            const auto stepEnd = sim_clock::now();
            const auto stepLength = stepEnd - stepStart;

            if(stepLength > dtDuration) {
                totalOverrunTime += (stepLength - dtDuration);
            }
            else {
                const auto sleepTime = dtDuration - stepLength;
                std::this_thread::sleep_for(sleepTime);
                totalSleepTime += sleepTime;
            }
        }
    }
#endif

    if (which == CPU){
        while (g_SignalStatus == 0)
        {
            // Get time step started at
            const auto stepStart = sim_clock::now();

            locust.runCPU(DT); // run next batch

#ifdef TIMING
            fprintf(timeros, "%f %f %f \n", neuron_tme, synapse_tme, learning_tme);
#endif
            locust.sum_spikes();
            locust.outputDN_spikes(osf2, which);

            const auto stepEnd = sim_clock::now();
            const auto stepLength = stepEnd - stepStart;

            if(stepLength > dtDuration) {
                totalOverrunTime += (stepLength - dtDuration);
            }
            else {
                const auto sleepTime = dtDuration - stepLength;
                std::this_thread::sleep_for(sleepTime);
                totalSleepTime += sleepTime;
            }
        }
    }
    timer.stopTimer();
    printf("Ran for %fms, overran by %fms and slept for %fms\n", t, double_ms(totalOverrunTime).count() , double_ms(totalSleepTime).count());
}

void cameraThreadHandler(unsigned int device)
{
    typedef std::chrono::duration<double> double_s;

    // Check camera has opened correctly
    cv::VideoCapture camera(device);
    if(!camera.isOpened()) {
        throw std::runtime_error("Cannot open camera");
    }


    // Get frame dimensions
    const unsigned int width = camera.get(CV_CAP_PROP_FRAME_WIDTH);
    const unsigned int height = camera.get(CV_CAP_PROP_FRAME_HEIGHT);

    const unsigned int margin = (width - height) / 2;
    const cv::Rect square(cv::Point(margin, 0), cv::Point(width - margin, height));

    // Read first frame so we can create ROI
    cv::Mat rawFrame;
    if(!camera.read(rawFrame)) {
        throw std::runtime_error("Cannot read first frame");
    }

    // Create ROI
    cv::Mat squareROI = rawFrame(square);
    cv::Mat greyscaleFrame;
    cv::Mat downsampledFrame;

    cv::namedWindow("Frame", CV_WINDOW_NORMAL);
    cv::resizeWindow("Frame", 320, 320);

    const auto cameraBegin = sim_clock::now();
    unsigned int i;
    for (i = 0; g_SignalStatus == 0; i++)
    {
        // Read frame
        if(!camera.read(rawFrame)) {
            throw std::runtime_error("Cannot read frame");
        }

        // Convert frame to greyscale
        cv::cvtColor(squareROI, greyscaleFrame, CV_BGR2GRAY);

        // Resample
        cv::resize(greyscaleFrame, downsampledFrame,
                   cv::Size(32, 32));

        cv::imshow("Frame", downsampledFrame);
        cv::waitKey(1);
    }

    const auto cameraEnd = sim_clock::now();
    const auto cameraTimeS = double_s(cameraEnd - cameraBegin);
    printf("%f FPS\n", (double)(i - 1) / cameraTimeS.count());
}
} // Anonymous namespace

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: classol_sim <basename> <CPU=0, GPU=1> \n");
        return 1;
    }
    int which= atoi(argv[2]);
    string OutDir = toString(argv[1]) +"_output";
    string name;
    name= OutDir+ "/"+ toString(argv[1]) + toString(".time");
    FILE *timef= fopen(name.c_str(),"a");

    patSetTime= (int) (PAT_TIME/DT);
    patFireTime= (int) (PATFTIME/DT);
    fprintf(stdout, "# DT %f \n", DT);
    fprintf(stdout, "# T_REPORT_TME %f \n", T_REPORT_TME);
    fprintf(stdout, "# SYN_OUT_TME %f \n",  SYN_OUT_TME);
    fprintf(stdout, "# PATFTIME %f \n", PATFTIME);
    fprintf(stdout, "# patFireTime %d \n", patFireTime);
    fprintf(stdout, "# PAT_TIME %f \n", PAT_TIME);
    fprintf(stdout, "# patSetTime %d \n", patSetTime);
    fprintf(stdout, "# TOTAL_TME %f \n", TOTAL_TME);

    name= OutDir+ "/"+ toString(argv[1]) + toString(".out.Vm");
    FILE *osf= fopen(name.c_str(),"w");
    name= OutDir+ "/"+ toString(argv[1]) + toString(".out.st");
    FILE *osf2= fopen(name.c_str(),"w");

#ifdef TIMING
    name= OutDir+ "/"+ toString(argv[1]) + toString(".timingprofile");
    FILE *timeros= fopen(name.c_str(),"w");
    double tme;
#endif

    //-----------------------------------------------------------------
    // build the neuronal circuitery
    classol locust;

#ifdef TIMING
    timer.startTimer();
#endif

    fprintf(stdout, "# reading PN-KC synapses ... \n");
    name= OutDir+ "/"+ toString(argv[1]) + toString(".pnkc");
    FILE *f= fopen(name.c_str(),"rb");
    locust.read_pnkcsyns(f);
    fclose(f);

#ifdef TIMING
    timer.stopTimer();
    tme= timer.getElapsedTime();
    fprintf(timeros, "%% Reading PN-KC synapses: %f \n", tme);
    timer.startTimer();
#endif

    fprintf(stdout, "# reading PN-LHI synapses ... \n");
    name= OutDir+ "/"+ toString(argv[1]) + toString(".pnlhi");
    f= fopen(name.c_str(), "rb");
    locust.read_pnlhisyns(f);
    fclose(f);

#ifdef TIMING
    timer.stopTimer();
    tme= timer.getElapsedTime();
    fprintf(timeros, "%% Reading PN-LHI synapses: %f \n", tme);
    timer.startTimer();
#endif
  
    fprintf(stdout, "# reading KC-DN synapses ... \n");
    name= OutDir+ "/"+ toString(argv[1]) + toString(".kcdn");
    f= fopen(name.c_str(), "rb");
    locust.read_kcdnsyns(f);

#ifdef TIMING
    timer.stopTimer();
    tme= timer.getElapsedTime();
    fprintf(timeros, "%% Reading KC-DN synapses: %f \n", tme);
    timer.startTimer();
#endif

    fprintf(stdout, "# reading input patterns ... \n");
    name= OutDir+ "/"+ toString(argv[1]) + toString(".inpat");
    f= fopen(name.c_str(), "rb");
    locust.read_input_patterns(f);
    fclose(f);

#ifdef TIMING
    timer.stopTimer();
    tme= timer.getElapsedTime();
    fprintf(timeros, "%% Reading input patterns: %f \n", tme);
    timer.startTimer();
#endif

    locust.generate_baserates();
#ifndef CPU_ONLY
    if (which == GPU) {
        locust.allocate_device_mem_patterns();
    }
#endif
    locust.init(which);         // this includes copying g's for the GPU version

#ifdef TIMING
    timer.stopTimer();
    tme= timer.getElapsedTime();
    fprintf(timeros, "%% Initialisation: %f \n", tme);
#endif

    fprintf(stdout, "# neuronal circuitery built, start computation ... \n\n");

    std::signal(SIGINT, signalHandler);

    // Start threads
    std::thread gennThread(gennThreadHandler, which, osf, osf2, std::ref(locust));
    std::thread cameraThread(cameraThreadHandler, 0);
    gennThread.join();
    cameraThread.join();

#ifndef CPU_ONLY
    if (which == GPU) pullDNStateFromDevice();
#endif
    cerr << "output files are created under the current directory." << endl;
    fprintf(timef, "%d %u %u %u %u %u %.4f %.2f %.1f %.2f\n",which, locust.model.getNumNeurons(), locust.sumPN, locust.sumKC, locust.sumLHI, locust.sumDN, timer.getElapsedTime(),VDN[0], TOTAL_TME, DT);
    fprintf(stdout, "GPU=%d, %u neurons, %u PN spikes, %u KC spikes, %u LHI spikes, %u DN spikes, simulation took %.4f secs, VDN[0]=%.2f DT=%.1f %.2f\n",which, locust.model.getNumNeurons(), locust.sumPN, locust.sumKC, locust.sumLHI, locust.sumDN, timer.getElapsedTime(),VDN[0], TOTAL_TME, DT);

    fclose(osf);
    fclose(osf2);
    fclose(timef);

#ifdef TIMING
    fclose(timeros);
#endif

#ifndef CPU_ONLY
    if (which == GPU) {
        locust.free_device_mem();
    }
#endif
    return 0;
}
