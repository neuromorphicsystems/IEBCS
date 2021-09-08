%% About definesimu_matlab.mlx
% This file defines the MATLAB interface to the library |simu_matlab|.
%
% Commented sections represent C++ functionality that MATLAB cannot automatically define. To include
% functionality, uncomment a section and provide values for &lt;SHAPE&gt;, &lt;DIRECTION&gt;, etc. For more
% information, see <matlab:helpview(fullfile(docroot,'matlab','helptargets.map'),'cpp_define_interface') Define MATLAB Interface for C++ Library>.



%% Setup. Do not edit this section.
function libDef = definesimu_matlab()
libDef = clibgen.LibraryDefinition("simu_matlabData.xml");
%% OutputFolder and Libraries 
libDef.OutputFolder = "/home/joubertd/Documents/Soft/Blender/Saccades/Imager/blender_to_event/matlab";
libDef.Libraries = "";

%% C++ class |SimuICNSMatlab| with MATLAB name |clib.simu_matlab.SimuICNSMatlab| 
SimuICNSMatlabDefinition = addClass(libDef, "SimuICNSMatlab", "MATLABName", "clib.simu_matlab.SimuICNSMatlab", ...
    "Description", "clib.simu_matlab.SimuICNSMatlab    Representation of C++ class SimuICNSMatlab."); % Modify help description values as needed.

%% C++ class constructor for C++ class |SimuICNSMatlab| 
% C++ Signature: SimuICNSMatlab::SimuICNSMatlab()
SimuICNSMatlabConstructor1Definition = addConstructor(SimuICNSMatlabDefinition, ...
    "SimuICNSMatlab::SimuICNSMatlab()", ...
    "Description", "clib.simu_matlab.SimuICNSMatlab    Constructor of C++ class SimuICNSMatlab."); % Modify help description values as needed.
validate(SimuICNSMatlabConstructor1Definition);

%% C++ class method |initSimu| for C++ class |SimuICNSMatlab| 
% C++ Signature: void SimuICNSMatlab::initSimu(uint16_t x,uint16_t y)
initSimuDefinition = addMethod(SimuICNSMatlabDefinition, ...
    "void SimuICNSMatlab::initSimu(uint16_t x,uint16_t y)", ...
    "MATLABName", "initSimu", ...
    "Description", "initSimu    Method of C++ class SimuICNSMatlab."); % Modify help description values as needed.
defineArgument(initSimuDefinition, "x", "uint16");
defineArgument(initSimuDefinition, "y", "uint16");
validate(initSimuDefinition);

%% C++ class method |initContrast| for C++ class |SimuICNSMatlab| 
% C++ Signature: void SimuICNSMatlab::initContrast(double_t c_pos_arg,double_t c_neg_arg,double_t c_noise_arg)
initContrastDefinition = addMethod(SimuICNSMatlabDefinition, ...
    "void SimuICNSMatlab::initContrast(double_t c_pos_arg,double_t c_neg_arg,double_t c_noise_arg)", ...
    "MATLABName", "initContrast", ...
    "Description", "initContrast    Method of C++ class SimuICNSMatlab."); % Modify help description values as needed.
defineArgument(initContrastDefinition, "c_pos_arg", "double");
defineArgument(initContrastDefinition, "c_neg_arg", "double");
defineArgument(initContrastDefinition, "c_noise_arg", "double");
validate(initContrastDefinition);

%% C++ class method |initLat| for C++ class |SimuICNSMatlab| 
% C++ Signature: void SimuICNSMatlab::initLat(double_t lat_arg,double_t jit_arg,double_t ref_arg,double_t tau_arg)
initLatDefinition = addMethod(SimuICNSMatlabDefinition, ...
    "void SimuICNSMatlab::initLat(double_t lat_arg,double_t jit_arg,double_t ref_arg,double_t tau_arg)", ...
    "MATLABName", "initLat", ...
    "Description", "initLat    Method of C++ class SimuICNSMatlab."); % Modify help description values as needed.
defineArgument(initLatDefinition, "lat_arg", "double");
defineArgument(initLatDefinition, "jit_arg", "double");
defineArgument(initLatDefinition, "ref_arg", "double");
defineArgument(initLatDefinition, "tau_arg", "double");
validate(initLatDefinition);

%% C++ class method |initNoise| for C++ class |SimuICNSMatlab| 
% C++ Signature: void SimuICNSMatlab::initNoise(double_t * pos_dist,double_t * neg_dist,size_t size)
%initNoiseDefinition = addMethod(SimuICNSMatlabDefinition, ...
%    "void SimuICNSMatlab::initNoise(double_t * pos_dist,double_t * neg_dist,size_t size)", ...
%    "MATLABName", "initNoise", ...
%    "Description", "initNoise    Method of C++ class SimuICNSMatlab."); % Modify help description values as needed.
%defineArgument(initNoiseDefinition, "pos_dist", "clib.array.simu_matlab.Double", "input", <SHAPE>); % '<MLTYPE>' can be clib.array.simu_matlab.Double, or double
%defineArgument(initNoiseDefinition, "neg_dist", "clib.array.simu_matlab.Double", "input", <SHAPE>); % '<MLTYPE>' can be clib.array.simu_matlab.Double, or double
%defineArgument(initNoiseDefinition, "size", "uint64");
%validate(initNoiseDefinition);

%% C++ class method |initImg| for C++ class |SimuICNSMatlab| 
% C++ Signature: void SimuICNSMatlab::initImg(double_t * img,size_t size)
%initImgDefinition = addMethod(SimuICNSMatlabDefinition, ...
%    "void SimuICNSMatlab::initImg(double_t * img,size_t size)", ...
%    "MATLABName", "initImg", ...
%    "Description", "initImg    Method of C++ class SimuICNSMatlab."); % Modify help description values as needed.
%defineArgument(initImgDefinition, "img", "clib.array.simu_matlab.Double", "input", <SHAPE>); % '<MLTYPE>' can be clib.array.simu_matlab.Double, or double
%defineArgument(initImgDefinition, "size", "uint64");
%validate(initImgDefinition);

%% C++ class method |updateImg| for C++ class |SimuICNSMatlab| 
% C++ Signature: void SimuICNSMatlab::updateImg(double_t * img,uint64_t dt,size_t size)
%updateImgDefinition = addMethod(SimuICNSMatlabDefinition, ...
%    "void SimuICNSMatlab::updateImg(double_t * img,uint64_t dt,size_t size)", ...
%    "MATLABName", "updateImg", ...
%    "Description", "updateImg    Method of C++ class SimuICNSMatlab."); % Modify help description values as needed.
%defineArgument(updateImgDefinition, "img", "clib.array.simu_matlab.Double", "input", <SHAPE>); % '<MLTYPE>' can be clib.array.simu_matlab.Double, or double
%defineArgument(updateImgDefinition, "dt", "uint64");
%defineArgument(updateImgDefinition, "size", "uint64");
%validate(updateImgDefinition);

%% C++ class method |getBufSize| for C++ class |SimuICNSMatlab| 
% C++ Signature: size_t SimuICNSMatlab::getBufSize()
getBufSizeDefinition = addMethod(SimuICNSMatlabDefinition, ...
    "size_t SimuICNSMatlab::getBufSize()", ...
    "MATLABName", "getBufSize", ...
    "Description", "getBufSize    Method of C++ class SimuICNSMatlab."); % Modify help description values as needed.
defineOutput(getBufSizeDefinition, "RetVal", "uint64");
validate(getBufSizeDefinition);

%% C++ class method |getBuffer| for C++ class |SimuICNSMatlab| 
% C++ Signature: void SimuICNSMatlab::getBuffer(uint64_t * ts,uint16_t * x,uint16_t * y,uint8_t * p,size_t size)
%getBufferDefinition = addMethod(SimuICNSMatlabDefinition, ...
%    "void SimuICNSMatlab::getBuffer(uint64_t * ts,uint16_t * x,uint16_t * y,uint8_t * p,size_t size)", ...
%    "MATLABName", "getBuffer", ...
%    "Description", "getBuffer    Method of C++ class SimuICNSMatlab."); % Modify help description values as needed.
%defineArgument(getBufferDefinition, "ts", "clib.array.simu_matlab.UnsignedLong", "input", <SHAPE>); % '<MLTYPE>' can be clib.array.simu_matlab.UnsignedLong, or uint64
%defineArgument(getBufferDefinition, "x", "clib.array.simu_matlab.UnsignedShort", "input", <SHAPE>); % '<MLTYPE>' can be clib.array.simu_matlab.UnsignedShort, or uint16
%defineArgument(getBufferDefinition, "y", "clib.array.simu_matlab.UnsignedShort", "input", <SHAPE>); % '<MLTYPE>' can be clib.array.simu_matlab.UnsignedShort, or uint16
%defineArgument(getBufferDefinition, "p", "clib.array.simu_matlab.UnsignedChar", "input", <SHAPE>); % '<MLTYPE>' can be clib.array.simu_matlab.UnsignedChar, or uint8
%defineArgument(getBufferDefinition, "size", "uint64");
%validate(getBufferDefinition);

%% C++ class method |destroy| for C++ class |SimuICNSMatlab| 
% C++ Signature: void SimuICNSMatlab::destroy()
destroyDefinition = addMethod(SimuICNSMatlabDefinition, ...
    "void SimuICNSMatlab::destroy()", ...
    "MATLABName", "destroy", ...
    "Description", "destroy    Method of C++ class SimuICNSMatlab."); % Modify help description values as needed.
validate(destroyDefinition);

%% C++ class method |setDebug| for C++ class |SimuICNSMatlab| 
% C++ Signature: void SimuICNSMatlab::setDebug()
setDebugDefinition = addMethod(SimuICNSMatlabDefinition, ...
    "void SimuICNSMatlab::setDebug()", ...
    "MATLABName", "setDebug", ...
    "Description", "setDebug    Method of C++ class SimuICNSMatlab."); % Modify help description values as needed.
validate(setDebugDefinition);

%% C++ class constructor for C++ class |SimuICNSMatlab| 
% C++ Signature: SimuICNSMatlab::SimuICNSMatlab(SimuICNSMatlab const & input1)
SimuICNSMatlabConstructor2Definition = addConstructor(SimuICNSMatlabDefinition, ...
    "SimuICNSMatlab::SimuICNSMatlab(SimuICNSMatlab const & input1)", ...
    "Description", "clib.simu_matlab.SimuICNSMatlab    Constructor of C++ class SimuICNSMatlab."); % Modify help description values as needed.
defineArgument(SimuICNSMatlabConstructor2Definition, "input1", "clib.simu_matlab.SimuICNSMatlab", "input");
validate(SimuICNSMatlabConstructor2Definition);

%% Validate the library definition
validate(libDef);

end
