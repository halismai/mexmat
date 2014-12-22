classdef TestClass < handle

  properties (SetAccess = private, Hidden = true)
    handle_; % class handle
  end

  methods
    function this = TestClass()
      this.handle_ = test_class_mex('new');
    end

    function delete(this)
      test_class_mex('delete', this.handle_);
    end

    function add(this, str)
      test_class_mex('add', this.handle_, str);
    end

    function printOut(this)
      test_class_mex('print', this.handle_);
    end

    function out = getStrings(this)
      out = test_class_mex('get', this.handle_);
    end

  end % methods

end % TestClass

