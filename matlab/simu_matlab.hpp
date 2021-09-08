/*
 * simu.hpp
 *
 *  Created on: 12 Feb 2021
 *      Author: joubertd
 */
#include "../cpp/simu.hpp"
#ifndef SIMU_MATLAB_HPP_
#define SIMU_MATLAB_HPP_
class SimuICNSMatlab{

    SimuICNS* simu_;
    std::vector<Event> buf_;
    uint8_t debug_;
public:
    SimuICNSMatlab(){
        debug_ = 0;
        std::cout<<"Init"<<std::endl;
    };
    void initSimu(uint16_t x, uint16_t y){
        simu_ = new SimuICNS(x, y);
    }
    void initContrast(double_t c_pos_arg, double_t c_neg_arg, double_t c_noise_arg){
        simu_->set_th(c_pos_arg, c_neg_arg, c_noise_arg);
        std::cout<<"initContrast"<<std::endl;
    }
    void initLat(double_t lat_arg, double_t jit_arg, double_t ref_arg, double_t tau_arg){
        simu_->set_lat(lat_arg, jit_arg, ref_arg, tau_arg);
        std::cout<<"initLat"<<std::endl;
    }
    void initNoise(double_t* pos_dist, double_t* neg_dist, size_t size){
        long int s = static_cast<uint16_t>(size / 72);
        simu_->init_noise(pos_dist, neg_dist, s);
        std::cout<<"initNoise"<<std::endl;
    }
    void initImg(double_t* img, size_t size){
        if(simu_->testShape(size)){
            simu_->init_img(img);
            std::cout<<"initImg"<<std::endl;
        }
    }
    void updateImg(double_t* img, uint64_t dt, size_t size){
        if(simu_->testShape(size)){
            buf_.clear();
            simu_->update_img(img, dt, buf_);
            std::cout<<"Image Updated"<<std::endl;
        }
    }
    size_t getBufSize(){return buf_.size();}
    void getBuffer(uint64_t* ts, uint16_t* x, uint16_t* y, uint8_t* p, size_t size){
        if(size == getBufSize()){
            for (auto i = 0; i < buf_.size(); i++){
                ts[i] = buf_.at(i).ts_;
                x[i] = buf_.at(i).x_;
                y[i] = buf_.at(i).y_;
                p[i] = buf_.at(i).p_;
            }
        }
    }; 
    void destroy(){delete simu_;};
    void setDebug(){
        debug_ = (debug_==1) ? 0 : 1;
        simu_->setDebug(debug_);};
};
#endif 
