library ieee;
use ieee.std_logic_1164.all;

entity D_latch is
port (
  d : in std_logic;
  clock : in in std_logic;

  q : out std_logic;
)

architecture Behavioral of D_latch is
begin
  process(d, clock) is
  begin
    if clock = '1' then
      q <= d;
    end if;
  end process;
end architecture Behavioral;