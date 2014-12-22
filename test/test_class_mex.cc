#include "mexmat.h"

#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <sstream>

//
// header for our simple class. It takes strings and converts to hex string
//
class StringToHex
{
 public:
  StringToHex() {}

 public:
  inline void add(const std::string& s)
  {
    _hex.push_back( to_hex(s) );
  }

  inline size_t size() const { return _hex.size(); }

  inline const std::vector<std::string>& data() const { return _hex; }

  inline void print(std::ostream& os = std::cout) const
  {
  	std::copy(std::begin(_hex), std::end(_hex), std::ostream_iterator<std::string>(os, "\n") );
  }

 private:
  std::vector<std::string> _hex;
  static const char* HEX;

  inline std::string to_hex(const std::string& str) const
  {
    std::string ret;

    for(auto c : str) {
      ret.push_back( HEX[ c>>4 ] );
      ret.push_back( HEX[ c&15 ] );
    }

    return ret;
  }
}; // StringToHex
const char* StringToHex::HEX = "0123456789ABCDEF";


//
// implementation of commands/things we can do with the class
//
void do_command_new(int nlhs, mxArray* plhs[],
                    int nrhs, mxArray const* prhs[])
{
  static const char* USAGE = "handle = fn('new')";

  mex::nargchk(1,1, nrhs, USAGE);
  mex::nargchk(1,1, nlhs, USAGE); // must have output

  plhs[0] = mex::PtrToMex<StringToHex>(new StringToHex);
}

void do_command_delete(int nlhs, mxArray* plhs[],
                       int nrhs, mxArray const* prhs[])
{
  static const char* USAGE = "fn('delete', handle)";

  mex::nargchk(2,2, nrhs, USAGE);
  mex::nargchk(0,0, nlhs, USAGE); // no output

  mex::DeleteClass<StringToHex>( prhs[1] );
}

void do_command_add(int nlhs, mxArray* plhs[],
                    int nrhs, mxArray const* prhs[])
{
  static const char* USAGE = "fn('delete', handle, 'string')";

  mex::nargchk(3,3, nrhs, USAGE);
  mex::nargchk(0,0, nlhs, USAGE); // no output

  auto ptr = mex::MexToPtr<StringToHex>( prhs[1] );
  ptr->add( mex::getString(prhs[2]) );

}

void do_command_print(int nlhs, mxArray* plhs[],
                      int nrhs, mxArray const* prhs[])
{
  static const char* USAGE = "fn('print', handle)";

  mex::nargchk(2,2,nrhs,USAGE);
  mex::nargchk(0,0,nlhs,USAGE);

  mex::MexToPtr<StringToHex>( prhs[1] )->print();
}

void do_command_get(int nlhs, mxArray* plhs[],
                    int nrhs, mxArray const* prhs[])
{
  static const char* USAGE = "out = fn('get', handle)";

  mex::nargchk(2,2,nrhs,USAGE);
  // mex::nargchk(1,1,nlhs,USAGE); // ok if no output
  auto ptr = mex::MexToPtr<StringToHex>( prhs[1] );
  const auto& data = ptr->data();

  mex::Cell ret( 1, ptr->size() );

  for(size_t i = 0; i < data.size(); ++i) {
    char* out = mex::calloc<char>( data[i].size() );
    memcpy(out, &data[i][0], data[i].size());
    ret.set(i, mxCreateString(out));
  }

  plhs[0] = ret.release();
}

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, mxArray const* prhs[])
{
  // first argument is a string of the command to perform
  if(nrhs < 1) {
    mex::error("must have at least one argument");
  }

  const std::string command = mex::getString( prhs[0] );

  if("new" == command)
    do_command_new(nlhs, plhs, nrhs, prhs);
  else if("delete" == command)
    do_command_delete(nlhs, plhs, nrhs, prhs);
  else if("add" == command)
    do_command_add(nlhs, plhs, nrhs, prhs);
  else if("print" == command)
    do_command_print(nlhs, plhs, nrhs, prhs);
  else if("get" == command)
    do_command_get(nlhs, plhs, nrhs, prhs);
  else {
    std::stringstream msg;
    msg << "unknown command '" << command << "'\n";
    mex::error(msg.str());
  }
}


