classdef ClassWrapperBase < handle
  % classdef ClassWrapperBase < handle

  properties (Access = private, Hidden = true)
    wrapper_fn_;
    hdle_;
  end

  properties (Constant = true, Hidden = true)
    Debug_ = false;
  end

  methods
    function this = ClassWrapperBase(w_fn, varargin)
      %function this = ClassWrapperBase(w_fn)
      %
      % INPUT
      %   w_fn   a string/function handle to the mex interface
      %

      if ischar(w_fn)
        this.wrapper_fn_ = str2func(w_fn);
      elseif ishandle(w_fn)
        this.wrapper_fn_ = w_fn;
      else
        error('input must be a function handle, or a function name');
      end

      this.hdle_ = this.wrapper_fn_('new', varargin{:});
    end

    function delete(this)

      if this.Debug_
        fprintf('delete class with handle %s\n', num2str(this.hdle_))
      end

      this.wrapper_fn_('delete', this.hdle_);
    end

  end % methods


  methods (Access = protected)
    function varargout = call_mex(this, command, varargin)
      % function varargout = call_mex(command, varargin)

      if(this.Debug_)
        fprintf('calling %s with %d out args %d in args\n', command, nargout, nargin);
      end

      [varargout{1:nargout}] = this.wrapper_fn_(command, this.hdle_, varargin{:});

    end
  end

end
