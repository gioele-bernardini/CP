library ieee;
use ieee.std_logic_1164.all;

entity MultibitRegister is
  generic (
    n : natural := 4;
  )
  port (
    d : in std_logic_vector(generic-1 downto 0);
    clock : in std_logic;
    set : in std_logic;
    reset : in std_logic;

    q : out std_logic_vector(generic-1 downto 0);
  )
end entity MultibitRegister;

architecture Behavioral of MultibitRegister is
begin
  process (clock, set, reset) is
  begin
    if reset = '0' then
      q <= (others => '0');
    elsif rising_edge(clock) then
      q <= d;
    end if;
  end process;
end architecture Behavioral;