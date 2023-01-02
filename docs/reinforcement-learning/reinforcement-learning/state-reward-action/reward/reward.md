# 奖励

- [返回上层目录](../state-reward-action.md)

关于奖励的认知，$R_s^a$是对在状态$s$下执行动作$a$时可得到的**期望即时收益**的声明式数学表示，不隐含程序上的命令和动作信息，不能从这个表示方法推导出其隐含了执行$a$立刻就能收到奖励$r$或者$r$与下一时刻的状态$s'$无关的信息。强化学习中有很多不同形式的简写符号，不同地方的符号也经常不一致，比如Sutton的强化学习教材中用$r(s,a)$表示期望即时收益，其实际上是对$r(s,a)=\sum_rr\sum_{s′}⁡p(s′,r|s,a)$的简写，虽然$r$与$s'$有关，但$s'$在特定环境下本身也是由$(s,a)$决定的（比如围棋），为了表达简明跳过$s'$用$r(s,a)$表示也没什么问题，另外书中也常仅用$R_{t+1}$表示$r(s_t,a_t)$，这时一个状态和动作变量都没用，看到公式时能结合语境理解其基本含义就可以，具体的表示法和用到的变量不用过于纠结。

在Sutton的原作中关于奖励的阐述是，奖励是离开状态$s$时得到的，表示奖励是所离开状态$s$的奖励，也就是上一个状态。$R_s^a$是一个期望形式的定义。

（1）对于经过$s$和$a$，能得到确定性的$s'$，则$Q$和$V$的转换公式可以是
$$
\begin{aligned}
q(s,a)&=R_{ss'}^a + \gamma v(s')\\
&=R_s^a + \gamma v(s')
\end{aligned}
$$
其中，为什么$R_{ss'}^a$可以写成$R_s^a$，是因为$s'$是由$(s,a)$决定的，比如围棋程序，所以没必要写$s'$了。

（2）对于存在环境状态转移概率的环境来说，经过$s$和$a$，得到的$s'$是不确定的，那么$Q$和$V$的转换公式为
$$
\begin{aligned}
q_{\pi}(s,a)&=\sum_{s'}P_{ss'}^a(R_{ss'}^a+\gamma v_{\pi}(s'))\\
&=\sum_{s'}P_{ss'}^a R_{ss'}^a + \gamma \sum_{s'}P_{ss'}^a v_{\pi}(s')\\
&=R_s^a + \gamma \sum_{s'}P_{ss'}^a v_{\pi}(s')
\end{aligned}
$$
看吧，即便存在下一个环境$s'$的奖励$R_{ss'}^a$，但是对于$(s,a)$来说，其奖励$R_s^a$是一个期望值，与你具体的下一个$s'$是什么没有关系，那这就可以认为在环境$s$下，一旦做出动作$a$，就马上能知道$(s,a)$的奖励$r$了，当然这只是概念上的。而实际上，这个$r$是怎么得来的呢，正如上式表示的那样，是一次次蒙特卡洛采样，对下一个环境$s'$中的得到的$(s,a)$的奖励$R_{ss'}^a$的平均（期望）值。

换句话说，可以理解为$R(s,a)$（即$R_s^a$）是一个和$s'$有关的随机变量，随机性来自于系统的状态转移概率。在对$s’$求期望后就得到了$R(s, a)$，是一个与$s'$无关的值。

再继续看上式，其中，$R_{ss'}^a$如果更严谨的说，其实也是个期望值，它的更详细的写法应该是：
$$
R_{ss'}^a=\sum_{r}\left[r\cdot p(r|s,a,s')\right]
$$
则
$$
\begin{aligned}
q_{\pi}(s,a)&=\sum_{s'}P_{ss'}^a\left(\sum_{r}\left(r\cdot p(r|s,a,s')\right)+\gamma v_{\pi}(s')\right)\\
&=\sum_{s'}P_{ss'}^a \sum_{r}\left(r\cdot p(r|s,a,s')\right) + \gamma \sum_{s'}P_{ss'}^a v_{\pi}(s')\\
&=\sum_{r}r\sum_{s'}\left(P_{ss'}^a \cdot p(r|s,a,s')\right) + \gamma \sum_{s'}P_{ss'}^a v_{\pi}(s')\\
&=\sum_{r}r\sum_{s'}p(s'|s,a)p(r|s,a,s') + \gamma \sum_{s'}P_{ss'}^a v_{\pi}(s')\\
&=\sum_{r}r\sum_{s'}\frac{p(s,a,s')}{p(s,a)}p(r|s,a,s') + \gamma \sum_{s'}P_{ss'}^a v_{\pi}(s')\ \text{可以省略，只是为了让你看的更明白}\\
&=\sum_{r}r\sum_{s'}\frac{p(r,s,a,s')}{p(s,a)} + \gamma \sum_{s'}P_{ss'}^a v_{\pi}(s')\ \text{可以省略，只是为了让你看的更明白}\\
&=\sum_{r}r\sum_{s'}p(s',r|s,a) + \gamma \sum_{s'}P_{ss'}^a v_{\pi}(s')\\
&=R_s^a + \gamma \sum_{s'}P_{ss'}^a v_{\pi}(s')
\end{aligned}
$$
上式中可以省略的那两行，依据其实是条件概率公式，即
$$
P(A|B)=\frac{P(AB)}{P(B)}
$$






