library IEEE;
use IEEE.std_logic_1164.all;

entity D_FF is
port (
  d : in std_logic;
  clock : in std_logic;
  -- Dovrebbe essere dichiarato due volte che reset e' usato solo in una architecture!
  reset : in std_logic;

  q : out std_logic;
)

architecture Behavioral of D_FF is
begin
  process(clock) is
  begin
    if clock = '1' then
      q <= d;
    end if;
  end process;
end architecture Behavioral;

architecture Behavioral of D_FF is
begin
  process(clock, reset)
  begin
    if reset = '1' then
      q <= '0';
    elsif rising_edge(clock) then
      q <= d;
    end if;
  end process;
end architecture Behavioral