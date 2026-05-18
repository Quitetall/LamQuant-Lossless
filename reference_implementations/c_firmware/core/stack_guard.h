#ifndef STACK_GUARD_H
#define STACK_GUARD_H

void stack_setup_hardware_trap(void);
void stack_overflow_exception(void);
void stack_reset_canaries(void);

#endif /* STACK_GUARD_H */
