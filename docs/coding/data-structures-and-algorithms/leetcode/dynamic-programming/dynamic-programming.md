# 动态规划

* [返回上层目录](../leetcode.md)





# 05最长回文子串

> 题目：给定一个字符串 `s`，找到 `s` 中最长的回文子串。你可以假设 `s` 的最大长度为 1000。
>
> **示例 1：**
>
> ```
> 输入: "babad"
> 输出: "bab"
> 注意: "aba" 也是一个有效答案。
> ```
>
> **示例 2：**
>
> ```
> 输入: "cbbd"
> 输出: "bb"
> ```

为了改进暴力法，我们首先观察如何避免在验证回文时进行不必要的重复计算。考虑`ababa`这个示例。如果我们已经知道`bab`是回文，那么很明显，`ababa`一定是回文，因为它的左首字母和右尾字母是相同的。

给出$P(i,j)$的定义如下：
$$
\begin{aligned}
P(i,j)=\left\{\begin{matrix}
&true, &\text{如果子串Si...Sj是回文子串}\\ 
&false, &\text{其他情况}
\end{matrix}\right.
\end{aligned}
$$
因此，
$$
P(i,j)=(P(i+1,j-1)\ \text{and}\ S_i==S_j)
$$
这产生了一个直观的动态规划解法，我们首先初始化一字母和二字母的回文，然后找到所有三字母回文，并依此类推…

复杂度分析

时间复杂度：$O(n^2)$，这里给出我们的运行时间复杂度为$O(n^2)$。

空间复杂度：$O(n^2)$，该方法使用$O(n^2)$的空间来存储表。

你能进一步优化上述解法的空间复杂度吗？

中心扩展算法
事实上，只需使用恒定的空间，我们就可以在$O(n^2)$的时间内解决这个问题。

我们观察到回文中心的两侧互为镜像。因此，回文可以从它的中心展开，并且只有2n - 1个这样的中心。

你可能会问，为什么会是2n - 1（即奇数n + 偶数n-1）个，而不是n个中心？原因在于所含字母数为偶数的回文的中心可以处于两字母之间（例如`abba`的中心在两个`b`之间）。

c++:

```c++
class Solution {
public:
	string longestPalindrome(string s) {
		if(s.empty()) return "";
		int start = 0, end = 0;
		for (int i = 0; i < s.length(); i++) {
			int len1 = expandAroundCenter(s, i, i);
			int len2 = expandAroundCenter(s, i, i + 1);
			int len = max(len1, len2);
			if (len > end - start + 1) {
				start = i - (len - 1) / 2;
				end = i + len / 2;
			}
		}
		return s.substr(start, end - start + 1);
	}
private:
	int expandAroundCenter(string s, int left, int right) {
		int L = left, R = right;
		while (L >= 0 && R < s.length() && s[L] == s[R]) {
			L--;
			R++;
		}
		return R - L - 1;
	}
};
```

[leetcode](https://leetcode-cn.com/problems/longest-palindromic-substring/solution/zui-chang-hui-wen-zi-chuan-by-leetcode/)



# 53最大子序和

> 题目：给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
>
> **示例:**
>
> ```js
> 输入: [-2,1,-3,4,-1,2,1,-5,4],
> 输出: 6
> 解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
> ```

动态规划法：
$$
dp[i]=max(dp[i-1]+nums[i], \ nums[i])
$$
c++:

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int dp = nums[0];
        int ret_max = dp;

        for(int i = 1; i < nums.size(); i++) {
            dp = max(dp + nums[i], nums[i]);
            ret_max = max(ret_max, dp);
        }

        return ret_max;
    }
};
```

[leetcode](https://leetcode-cn.com/problems/maximum-subarray/)



# 62不同路径

> 题目：一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。问总共有多少条不同的路径？
>

![leetcode-62](pic/leetcode-62.png)

例如，上图是一个7 x 3 的网格。有多少可能的路径？

说明：m 和 n 的值均不超过 100。

示例 1:

```js
输入: m = 3, n = 2
输出: 3
```

解释:

从左上角开始，总共有 3 条路径可以到达右下角。

```js
1. 向右 -> 向右 -> 向下
2. 向右 -> 向下 -> 向右
3. 向下 -> 向右 -> 向右
```

**思路一：排列组合**

因为机器到底右下角，向下几步，向右几步都是固定的，

比如，m=3, n=2，我们只要向下 1 步，向右 2 步就一定能到达终点。

所以有$C_{m+n-2}^{m-1}$。

**思路二：动态规划**

我们令`dp[i][j]`是到达 i, j 最多路径

动态方程：`dp[i][j] = dp[i-1][j] + dp[i][j-1]`。

注意，对于第一行`dp[0][j]`，或者第一列`dp[i][0]`，由于都是在边界，所以只能为1。

时间复杂度：O(m*n)

空间复杂度：O(m * n)

c++:

```c++
class Solution {
public:
	int uniquePaths(int m, int n) {
		int dp[m][n];
		for (int i = 0; i < n; i++) dp[0][i] = 1;
		for (int i = 0; i < m; i++) dp[i][0] = 1;
		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
			}
		}
		return dp[m - 1][n - 1];
	}
};
```

[leetcode](https://leetcode-cn.com/problems/unique-paths/)



# 63不同路径II

> 题目：一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。现在考虑**网格中有障碍物**。那么从左上角到右下角将会有多少条不同的路径？
>
> 网格中的障碍物和空位置分别用 `1` 和 `0` 来表示。

![leetcode-62](pic/leetcode-62.png)

这一题是不同路径的一个进阶版，添加了障碍物，同样采用动态规划，区别是：

* 对于第一列或者第一行，只要当前的格子上有障碍物，那么之后的列或者行均不可能到达。
* 对于中间的格子，当有障碍物时，该格子的路径设置为0，这样该格子对于其后续的格子的路径数就没有贡献了。

c++:

```c++
class Solution {
public:
	int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
		int m = (int)obstacleGrid.size();
		int n = (int)obstacleGrid[0].size();
		if(m < 1 || n < 1 || obstacleGrid[0][0] == 1) return 0;

		long dp[m][n];
		memset(dp, 0, m * n * sizeof(long));

		for (int i = 0; i < n; i++) {
			if(obstacleGrid[0][i] == 1) {
				dp[0][i] = 0;
				break;
			} else {
				dp[0][i] = 1;
			}
		}
		for (int i = 0; i < m; i++) {
			if(obstacleGrid[i][0] == 1) {
				dp[i][0] = 0;
				break;
			} else {
				dp[i][0] = 1;
			}
		}
		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				if(obstacleGrid[i][j] == 1) {
					dp[i][j] = 0;
				} else {
					dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
				}
			}
		}
		return dp[m - 1][n - 1];
	}
};
```

[leetcode](https://leetcode-cn.com/problems/unique-paths-ii/solution/dong-tai-gui-hua-duo-yu-yan-zong-you-yi-kuan-gua-h/)



# 64最小路径和

> 题目：给定一个包含非负整数的 *m* x *n* 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
>
> **说明：**每次只能向下或者向右移动一步。

![leetcode-62](pic/leetcode-62.png)

**示例:**

```js
输入:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 7
解释: 因为路径 1→3→1→1→1 的总和最小。
```

动态方程：`dp[i][j] = min(dp[i-1][j] + dp[i][j-1]) + grid[i][j]`。

c++:

```c++
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        if(m < 1 || n < 1) return 0;
        if(m == 1 && n == 1) return grid[0][0];

        int sum[m][n];
        sum[0][0] = grid[0][0];

        for(int i = 1; i < m; i++) {
            sum[i][0] = sum[i-1][0] + grid[i][0];
        }

        for(int j = 1; j < n; j++) {
            sum[0][j] = sum[0][j-1] + grid[0][j];
        }

        for(int i = 1; i < m; i++) {
            for(int j = 1; j < n; j++) {
                sum[i][j] = min(sum[i-1][j], sum[i][j-1]) + grid[i][j];
            }
        }

        return sum[m-1][n-1];
    }
};
```

[leetcode](https://leetcode-cn.com/problems/minimum-path-sum/)



# 70爬楼梯

> 题目：假设你正在爬楼梯。需要 *n* 阶你才能到达楼顶。
>
> 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
>
> **注意：**给定 *n* 是一个正整数。

**示例1:**

```js
输入： 2
输出： 2
解释： 有两种方法可以爬到楼顶。
1.  1 阶 + 1 阶
2.  2 阶
```

**示例2:**

```js
输入： 3
输出： 3
解释： 有三种方法可以爬到楼顶。
1.  1 阶 + 1 阶 + 1 阶
2.  1 阶 + 2 阶
3.  2 阶 + 1 阶
```

动态方程：`dp[i] = dp[i-1] + dp[i-2]`。

c++:

```c++
class Solution {
public:
    int climbStairs(int n) {
        if(n <= 0) 
            return 0;
        else if(n == 1) 
            return 1;
        else if(n == 2) 
            return 2;
        
        int back2 = 1;
        int back1 = 2;
        int cur;
        for(int i = 3; i <= n; i++) {
            cur = back1 + back2;
            back2 = back1;
            back1 = cur;
        }

        return cur;
    }
};
```

[leetcode](https://leetcode-cn.com/problems/climbing-stairs/)

注意，不能用递归，会超出时间限制：

```c++
class Solution {
public:
    int climbStairs(int n) {
        if(n <= 0) 
            return 0;
        else if(n == 1) 
            return 1;
        else if(n == 2) 
            return 2;
        
        return climbStairs(n - 1) + climbStairs(n - 2);
    }
};
```

为什么？自己想一下（展开是一个树，会有很多重复计算）。



# 91解码方法

> 题目：一条包含字母 `A-Z` 的消息通过以下方式进行了编码：
>
> ```
> 'A' -> 1
> 'B' -> 2
> ...
> 'Z' -> 26
> ```
>
> 给定一个只包含数字的**非空**字符串，请计算解码方法的总数。

**示例1:**

```js
输入: "12"
输出: 2
解释: 它可以解码为 "AB"（1 2）或者 "L"（12）。
```

**示例2:**

```js
输入: "226"
输出: 3
解释: 它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。
```

动态方程：`dp[i] = dp[i-1] + dp[i-2]`。

本体利用动态规划比较容易解决，但需要注意分情况讨论

* dp[i]为str[0,...,i]的译码方法综述

* 分情况讨论：（建立最优子结构）

  * 若s[i]='0'，那么若s[i-1]='1'或者'2'，则dp[i]=dp[i-2]；否则return 0

    解释：s[i-1]+s[i]唯一被译码，不增加情况

  * 若s[i-1]='1'，则dp[i]=dp[i-1]+dp[i-2]

    解释：S[i-1]与s[i]分开译码，为dp[i-1]；合并译码，为dp[i-2]

  * 若s[i-1]=‘2’且'1'<=s[i]<='6'，则dp[i]=dp[i-1]+dp[i-2]

    解释：同上

* 由分析可知，dp[i]仅可能与前两项有关，故可以用单变量代替dp[]数组，将空间复杂度从O(n)降低到O(1)。

c++:

```c++
class Solution {
public:
    int numDecodings(string s) {
        if (s[0] == '0') return 0;
        int pre2 = 1, pre1 = 1, curr = 1;
        for (int i = 1; i < s.size(); i++) {
            if (s[i] == '0')
                if (s[i - 1] == '1' || s[i - 1] == '2') curr = pre2;
                else return 0;
            else if (s[i - 1] == '1' || (s[i - 1] == '2' && s[i] >= '1' && s[i] <= '6'))
                curr = pre1 + pre2;
            pre2 = pre1;
            pre1 = curr;
        }
        return curr;
    }
};
```

[leetcode](https://leetcode-cn.com/problems/decode-ways/)

如果觉得上述代码不好理解的话，请看下面我实际不断解决错误调试出来的代码：

```c++
class Solution {
public:
    int numDecodings(string s) {
        if(s.empty()) return 0;
        if(s[0] == '0') return 0;
        int pre2 = 1, pre1 = 1, cur = 1;

        for(int i = 1; i < s.size(); i++) {
            if(s[i] == '0') {
                if(s[i - 1] == '1' || s[i - 1] == '2') {
                    cur = pre2;
                } else {
                    cur = 0;
                    break;
                }
            } else if(s[i - 1] == '2') {
                if(s[i] >= '0' && s[i] <= '6') {
                    cur = pre2 + pre1;
                } else {
                    cur = pre1;
                }
            } else if(s[i - 1] <= '0' || s[i - 1] >= '3') {
                cur = pre1;
            } else {
                cur = pre2 + pre1;
            }
            pre2 = pre1;
            pre1 = cur;
        } 
        return cur;
    }
};
```



# 96.不同的二叉搜索树

> 题目：给定一个整数 *n*，求以 1 ... *n* 为节点组成的二叉搜索树有多少种？

**示例1:**

```js
输入: 3
输出: 5
解释:
给定 n = 3, 一共有 5 种不同结构的二叉搜索树:

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3
```

假设$n$个节点存在二叉排序树的个数是$G(n)$，令$f(i)$为以$i$为根的二叉搜索树的个数，则
$$
G(n)=f(1)+f(2)+f(3)+f(4)+...+f(n)
$$
当$i$为根节点时，其左子树节点个数为$i-1$个，右子树节点为$n-i$，则
$$
f(i)=G(i−1)∗G(n−i)
$$
综合两个公式可以得到 卡特兰数公式
$$
G(n)=G(0)∗G(n−1)+G(1)∗(n−2)+...+G(n−1)∗G(0)
$$
c++:

```c++
class Solution {
public:
    int numTrees(int n) {
        int dp[n+1];
        memset(dp, 0, (n + 1) * sizeof(int));
        dp[0] = 1;
        dp[1] = 1;

        for(int i = 2; i <= n; i++) {
            for(int j = 0; j <= i - 1; j++) {
                dp[i] += dp[j] * dp[i - j - 1];
            }
        }

        return dp[n];
    }
};
```

[leetcode](https://leetcode-cn.com/problems/unique-binary-search-trees/)



# 95. 不同的二叉搜索树 II

> 题目：给定一个整数 *n*，生成所有由 1 ... *n* 为节点所组成的**二叉搜索树**。

**示例1:**

```js
输入: 3
输出:
[
  [1,null,3,2],
  [3,2,null,1],
  [3,1,null,null,2],
  [2,1,3],
  [1,null,2,null,3]
]
解释:
以上的输出对应以下 5 种不同结构的二叉搜索树：

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3
注意：实际只需要输出每个可能的树的根结点即可。
```

c++:

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<TreeNode*> helper(int start,int end){
        vector<TreeNode*> ret;
        if(start > end)
            ret.push_back(nullptr);
        
        for(int i=start;i<=end;i++){
            vector<TreeNode*> left = helper(start,i-1);
            vector<TreeNode*> right = helper(i+1,end);
            for(auto l : left){
                for(auto r : right){
                    TreeNode* root = new TreeNode(i);
                    root -> left = l;
                    root -> right = r;
                    ret.push_back(root);
                }
            }
        }
        return ret;
    }
    
    vector<TreeNode*> generateTrees(int n) {
        vector<TreeNode*> ret;
        if(n == 0)
            return ret;    
        ret = helper(1,n);
        return ret;
    }
};
```

[leetcode](https://leetcode-cn.com/problems/unique-binary-search-trees-ii/)



# 120. 三角形最小路径和

> 题目：给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。
>
> 例如，给定三角形：
>
> ```js
> [
>      [2],
>     [3,4],
>    [6,5,7],
>   [4,1,8,3]
> ]
> ```
>
> 自顶向下的最小路径和为 `11`（即，**2** + **3** + **5** + **1** = 11）。

c++:

```c++
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        // 动态规划, 自底向上  
        // 递推式 dp[i][j] = min(dp[i+1][j], dp[i+1][j+1]) + triangle[i][j];
        int rowSize = triangle.size();
        vector<vector<int>> dp = triangle;
                
        for(int i = rowSize - 2;i >= 0; i--) {        
            for(int j = 0; j < triangle[i].size(); j++) {
                dp[i][j] = min(dp[i+1][j], dp[i+1][j+1]) + triangle[i][j];
            }
        }
        return dp[0][0];
    }
};
```

[leetcode](https://leetcode-cn.com/problems/triangle/)



# 121. 买卖股票的最佳时机

> 题目：给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
>
> 如果你最多只允许完成一笔交易（即买入和卖出一支股票），设计一个算法来计算你所能获取的最大利润。
>
> 注意你不能在买入股票前卖出股票。

示例 1:

```
输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
```

示例 2:

```
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
```

c++:

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int max = 0, sum = 0;
        for(int i = 0; i < (int)prices.size() - 1; i++) {
            int delta = prices[i + 1] - prices[i];
            sum = sum + delta > delta? sum + delta: delta;
            if(sum > max) {
                max = sum;
            }
        }
        return max;
    }
};
```

[leetcode](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)



# 139. 单词拆分

> 题目：给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。
>
> 说明：
>
> * 拆分时可以重复使用字典中的单词。
> * 你可以假设字典中没有重复的单词。

示例 1:

```
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。
```

示例 2:

```
输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
     注意你可以重复使用字典中的单词。
```

利用string.compare来对字符串进行比较

遍历wordDict中的word，与s的字串（从i开始，往前数word.size()个）进行比较，如果比较结果一致，则查看dp。

i处dp[i]设置为1的条件是：从i往前数word.size()个，再往前数1个，如果该处dp也为1,则设置dp[i]=1

最后只需返回dp[s.size()]即可。

总结：该解法比较简单，常规DP解法。比较绕的地方是s的索引i和dp数组的索引之间的关系。

c++:

```c++
class Solution {
public:
	bool wordBreak(string s, vector<string>& wordDict) {
		vector<int> dp(s.size()+1, 0);
		dp[0] = 1;
		for(int i=0; i<s.size(); ++i) {
			for(auto word: wordDict) {
				int ws = word.size();
				if(i + 1 - ws >= 0) {
					int cur = s.compare(i + 1 - ws, ws, word);
					if (cur==0 && dp[i + 1 - ws]==1) {
						dp[i + 1] = 1;
					}
				}
			}
		}
		return dp[s.size()];
	}
};
```

[leetcode](https://leetcode-cn.com/problems/word-break/)



# 152. 乘积最大子序列

> 题目：给定一个整数数组 `nums` ，找出一个序列中乘积最大的连续子序列（该序列至少包含一个数）。

示例 1:

```
输入: [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。
```

示例 2:

```
输入: [-2,0,-1]
输出: 0
解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。
```

解题思路

这题是求数组中子区间的最大乘积，对于乘法，我们需要注意，负数乘以负数，会变成正数，所以解这题的时候我们需要维护两个变量，当前的最大值，以及最小值，最小值可能为负数，但没准下一步乘以一个负数，当前的最大值就变成最小值，而最小值则变成最大值了。

我们的动态方程可能这样：

```
maxDP[i + 1] = max(maxDP[i] * A[i + 1], A[i + 1],minDP[i] * A[i + 1])
minDP[i + 1] = min(minDP[i] * A[i + 1], A[i + 1],maxDP[i] * A[i + 1])
dp[i + 1] = max(dp[i], maxDP[i + 1])
```

这里，我们还需要注意元素为0的情况，如果A[i]为0，那么maxDP和minDP都为0，
我们需要从A[i + 1]重新开始。

c++:

```c++
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int size = nums.size();
        if(size == 0) return 0;

        int dp_max = nums[0];
        int dp_min = nums[0];
        int max_val = nums[0];
        for(int i = 1; i < size; i++) {
            int temp = dp_max;
            dp_max = max(max(dp_max * nums[i], nums[i]), dp_min * nums[i]);
            dp_min = min(min(temp * nums[i], nums[i]), dp_min * nums[i]);
            max_val = max(max_val, dp_max);
        }

        return max_val;
    }
};
```

[leetcode](https://leetcode-cn.com/problems/maximum-product-subarray/)



# 参考资料

* [动态规划](https://leetcode-cn.com/tag/dynamic-programming/)

本文参考此资料。

