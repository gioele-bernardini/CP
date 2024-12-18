library ieee;
use ieee.std_logic_1164.all;

entity D_FF is
generic (
  n : natural := 4;
)
port (
  D : in std_logic_vector(n-1 downto 0);
  clock : in std_logic;
  load_enable : in std_logic;

  Q : out std_logic_vector(n-1 downto 0);
)

architecture Behavioral of D_FF is
begin
  -- Sbagliato! rising_edge funziona a 4 mani con process!
  if rising_edge(clock) then
    if load_enable = '1' then
      D <= Q;
    end if;
  end if;
end architecture Behavioral;