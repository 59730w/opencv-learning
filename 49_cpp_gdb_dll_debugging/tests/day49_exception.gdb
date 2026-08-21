set pagination off
set confirm off
break 'std::__throw_out_of_range_fmt(char const*, ...)'
run assets/day45_input.jpg 9
backtrace
frame 3
info args
info locals
continue
