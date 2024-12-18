library ieee;
use ieee.std_logic_1164.all;

-- Serial In Serial Out (SISO)
entity siso is
port (
  d : in std_logic;
  clock : in std_logic;

  q : out std_logic;
)

architecture Behavioral of siso is
  signal reg : std_logic_vector(3 downto 0);
begin
  process(clock) is
  begin
    if rising_edge(clock) then
      -- Sicuro?
      reg(3) <= q;
      reg(2) <= reg(3);
      reg(1) <= reg(2);
      reg(0) <= reg(1);
    end if;
  end process;
  -- Cosa manca?
end architecture Behavioral;