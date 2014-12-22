# Last modified: Wed 18 Sep 2013 02:52:05 PM EDT
MAKEFILES = $(shell find . -path ./build -prune -o -name Makefile)
TOP_DIR = $(PWD)
$(info "directory $(TOP_DIR))


all:
	for f in $(MAKEFILES); do 	\
	  d=`dirname $$f`; 		\
	  if [ $$d != "." ]; then 	\
		cd "$$d" && make; 	\
		cd $(TOP_DIR); 		\
	  fi 				\
	done

.PHONY: all clean distclean doxy

# If makefile changes, maybe the list of sources has changed, so update doxygens list
change_doxy:
	sed -i '/INPUT.*=/s/^.*$$/INPUT = . /' Doxyfile

doxy:   change_doxy
	doxygen Doxyfile

clean:
	for f in $(MAKEFILES); do 	\
	  d=`dirname $$f`; 		\
	  if [ $$d != "." ]; then 	\
		cd "$$d" && make clean;	\
		cd $(TOP_DIR); 		\
	  fi 				\
	done
