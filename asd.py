#!/usr/bin/env python3

from pwn import *

context.loglevel = 'DEBUG'

io = process('./challenge')
# io = remote('domain', 'port')

# Challenge 1
print(io.recvline())
print(io.recvline())
print(io.recvline())
print(io.recvline())
print(io.recvline())
print(io.recvline())

io.sendline(b'I obey your orders')

# Challenge 2
data = io.recvuntil(b'?')
numbers = data.split()[-3::2]
number_1 = int(numbers[0])
number_2 = int(numbers[1][0:-1])
io.sendline(f'{number_1 + number_2}').encode()

# Challenge 3
io.sendlineafter(b'0xdeadbeef!', p64(0xdeadbeef))

io.interactive()  # Prevents from exit



# Cosa diversa
context.terminal = ['tmux', 'splitw', '-h']
io = process('./challenge')

gdb.attach(io, ''''
    bp *main+556
    c
           ''')

# E metto un input('WAIT') per andare alla schermata di gdb!

hexdump $rsi
