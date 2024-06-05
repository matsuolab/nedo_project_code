CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color
CPPFLAGS += -I/storage3/miniconda3/envs/.ds0_14_2/include/python3.11 -I/storage3/miniconda3/envs/.ds0_14_2/lib/python3.11/site-packages/pybind11/include 
LIBNAME = helpers
LIBEXT = .cpython-311-x86_64-linux-gnu.so

default: $(LIBNAME)$(LIBEXT)

%$(LIBEXT): %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
