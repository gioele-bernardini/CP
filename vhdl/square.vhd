library IEEE;
use IEEE.STR_LOGIC_1164.ALL;

endity quadrato is
port (
  a: in std_logic_vector(2 downto 0);
  s: out std_logic_vector(5 downto 0);
);
end quadrato;

architecture Behavioral of quadrato is
begin
  s(5) <= '0';
  s(4) <= 